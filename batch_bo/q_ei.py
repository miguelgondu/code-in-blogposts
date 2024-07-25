from pathlib import Path
import sys
from typing import Literal

from jax import config

config.update("jax_enable_x64", True)

sys.path.append(str(Path(__file__).resolve().parent))

from poli.repository import ToyContinuousBlackBox

import numpy as np
from scipy.stats.qmc import Sobol

import gpjax as gpx
from jax import grad, jit
import jax.numpy as jnp
import jax.random as jr
import optax as ox

from gpjax.decision_making.search_space import ContinuousSearchSpace
from gpjax.decision_making.utility_maximizer import (
    ContinuousSinglePointUtilityMaximizer,
)
from gpjax.decision_making.utility_functions.probability_of_improvement import (
    ProbabilityOfImprovement,
)

import seaborn as sns
import matplotlib.pyplot as plt

from guiding_example import (
    plot_array,
    fit_gp,
    plot_predicted_mean,
    plot_cummulative_regret,
)

from expected_improvement import ExpectedImprovement

THIS_DIR = Path(__file__).resolve().parent
FIG_DIR = THIS_DIR / "figures" / "q_ei"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def q_ei_loop(
    posterior: gpx.Module,
    dataset: gpx.Dataset,
    key: jr.PRNGKey,
    batch_size: int = 6,
    heuristic: Literal["kriging_believer", "constant_liar"] = "constant_liar",
    axes: list[plt.Axes] = None,
    iteration: int = 0,
) -> list[np.ndarray]:
    negative_dataset = gpx.Dataset(
        X=dataset.X,
        y=-dataset.y,
    )

    domain = np.array(
        [[x, y] for x in np.linspace(-10, 10, 100) for y in np.linspace(-10, 10, 100)]
    )

    batch_X = []
    batch_y = []
    for b in range(batch_size):
        # Fit a negative posterior
        negative_posterior = fit_gp(negative_dataset, key)
        # sample the acquisition function
        ei_factory = ExpectedImprovement()

        ei = ei_factory.build_utility_function(
            posteriors={"OBJECTIVE": negative_posterior},
            datasets={"OBJECTIVE": negative_dataset},
            key=key,
        )
        acq_values = ei(domain)

        next_element_of_batch = domain[np.argmax(acq_values)]

        batch_X.append(next_element_of_batch)
        if heuristic.lower() == "constant_liar":
            batch_y.append(min(dataset.y) - 1)
        elif heuristic.lower() == "kriging_believer":
            latent_dist = posterior(next_element_of_batch, dataset)
            batch_y.append(latent_dist.mean())
        else:
            raise ValueError("Unknown heuristic")

        negative_dataset = negative_dataset + gpx.Dataset(
            X=next_element_of_batch.reshape(1, -1), y=-batch_y[-1].reshape(1, 1)
        )

        if axes is not None:
            for ax in axes:
                ax.axis("off")

            plot_array(
                axes[b],
                x=np.linspace(-10, 10, 100),
                y=np.linspace(-10, 10, 100),
                array=acq_values.reshape(100, 100).T,
                vmin=None,
                vmax=None,
            )
            axes[b].scatter(
                next_element_of_batch[0],
                next_element_of_batch[1],
                color="red",
                s=100,
                marker="x",
            )

            fig = axes[b].get_figure()
            fig.savefig(FIG_DIR / f"q_ei_{batch_size*iteration + b}.jpg")

    return batch_X


def q_ei(
    num_iterations: int,
    batch_size: int = 6,
    heuristic: Literal["kriging_believer", "constant_liar"] = "constant_liar",
):
    # Starting with the same dataset as in the guiding example
    sns.set_theme(style="darkgrid", font_scale=2.0)
    function_name = "cross_in_tray"
    n_dims = 2
    f = ToyContinuousBlackBox(function_name=function_name, n_dimensions=n_dims)

    # Sobol sampling
    seed = 0
    sobol_sampler = Sobol(d=2, scramble=True, seed=seed)
    key = jr.PRNGKey(seed)
    key, subkey = jr.split(key)

    samples = sobol_sampler.random(n=10).reshape(-1, 2)
    samples = samples * (10 - (-10)) - 10
    noisy_evaluations = f(samples)  # + 0.25 * jr.normal(subkey, shape=(10, 1))

    dataset = gpx.Dataset(X=samples, y=noisy_evaluations)

    domain = np.array(
        [[x, y] for x in np.linspace(-10, 10, 100) for y in np.linspace(-10, 10, 100)]
    )

    for iteration in range(num_iterations):
        fig, axes = plt.subplot_mosaic(
            mosaic=[
                [
                    "predicted_mean",
                    "predicted_mean",
                    "sample_0",
                    "sample_1",
                    "sample_2",
                ],
                [
                    "predicted_mean",
                    "predicted_mean",
                    "sample_3",
                    "sample_4",
                    "sample_5",
                ],
                [
                    "cummulative_regret",
                    "cummulative_regret",
                    "cummulative_regret",
                    "cummulative_regret",
                    "cummulative_regret",
                ],
            ],
            height_ratios=[2, 2, 1.5],
            figsize=(5 * 5, 5 * 2.05),
        )
        key = jr.PRNGKey(iteration)

        # Fit the GP with the dataset we have so far
        print(dataset)
        print(dataset.X.shape)
        opt_posterior = fit_gp(dataset, key)

        # Predictive mean
        plot_predicted_mean(
            axes["predicted_mean"],
            dataset=dataset,
            posterior=opt_posterior,
            f=f,
        )
        axes["predicted_mean"].axis("off")

        plot_cummulative_regret(
            axes["cummulative_regret"],
            dataset,
            f,
            total_budget=10 + batch_size * num_iterations + 1,
        )

        batch_to_evaluate = q_ei_loop(
            posterior=opt_posterior,
            dataset=dataset,
            key=key,
            batch_size=batch_size,
            heuristic=heuristic,
            axes=[
                axes["sample_0"],
                axes["sample_1"],
                axes["sample_2"],
                axes["sample_3"],
                axes["sample_4"],
                axes["sample_5"],
            ],
            iteration=iteration,
        )
        axes["sample_1"].set_title(f"qEI - {heuristic}")
        batch_ = np.vstack(batch_to_evaluate)

        y = f(batch_)

        # Add the new evaluations to the dataset
        new_dataset = gpx.Dataset(X=batch_, y=y)
        dataset += new_dataset

    fig, axes = plt.subplot_mosaic(
        mosaic=[
            [
                "predicted_mean",
                "predicted_mean",
                "sample_0",
                "sample_1",
                "sample_2",
            ],
            [
                "predicted_mean",
                "predicted_mean",
                "sample_3",
                "sample_4",
                "sample_5",
            ],
            [
                "cummulative_regret",
                "cummulative_regret",
                "cummulative_regret",
                "cummulative_regret",
                "cummulative_regret",
            ],
        ],
        height_ratios=[2, 2, 1.5],
        figsize=(5 * 5, 5 * 2.05),
    )
    key = jr.PRNGKey(iteration)

    # Fit the GP with the dataset we have so far
    # print(dataset)
    # print(dataset.X.shape)
    opt_posterior = fit_gp(dataset, key)

    for ax in [
        axes["sample_0"],
        axes["sample_1"],
        axes["sample_2"],
        axes["sample_3"],
        axes["sample_4"],
        axes["sample_5"],
    ]:
        ax.axis("off")

    # Predictive mean
    plot_predicted_mean(
        axes["predicted_mean"],
        dataset=dataset,
        posterior=opt_posterior,
        f=f,
    )
    axes["predicted_mean"].axis("off")

    plot_cummulative_regret(
        axes["cummulative_regret"],
        dataset,
        f,
        total_budget=10 + batch_size * num_iterations + 1,
    )
    fig.savefig(FIG_DIR / f"q_ei_{batch_size*(iteration+1)}.jpg")


if __name__ == "__main__":
    q_ei(num_iterations=10, batch_size=6, heuristic="constant_liar")
