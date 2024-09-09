from pathlib import Path
import sys

from jax import config

from poli.repository import ToyContinuousBlackBox

import numpy as np
from scipy.stats.qmc import Sobol

import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr

from gpjax.gps import ConjugatePosterior

import seaborn as sns
import matplotlib.pyplot as plt

from guiding_example import (
    plot_array,
    fit_gp,
    plot_predicted_mean,
    plot_cummulative_regret,
)

config.update("jax_enable_x64", True)

sys.path.append(str(Path(__file__).resolve().parent))

THIS_DIR = Path(__file__).resolve().parent
FIG_DIR = THIS_DIR / "figures" / "gpbucb"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def build_ucb(
    dataset: gpx.Dataset,
    posterior: ConjugatePosterior,
    beta: float = 1.0,
) -> jnp.ndarray:
    def ucb(domain: jnp.ndarray):
        dist_ = posterior.predict(domain, train_data=dataset)
        mean, std = dist_.mean(), dist_.stddev()
        return mean + std * beta

    return ucb


def propose_batch(
    dataset: gpx.Dataset,
    posterior: gpx.Module,
    key: jr.PRNGKey,
    beta: float = 1.0,
    batch_size: int = 6,
    axes: list[plt.Axes] = None,
    iteration: int = 0,
) -> jnp.ndarray:
    batch_dataset = gpx.Dataset(X=dataset.X, y=dataset.y)

    batch_ = []
    domain = np.array(
        [[x, y] for x in np.linspace(-10, 10, 100) for y in np.linspace(-10, 10, 100)]
    )
    for b in range(batch_size):
        # Maximize the UCB
        ucb = build_ucb(batch_dataset, posterior, beta=beta)

        ucb_values = ucb(domain)

        # Find the maximum value of the UCB
        max_ucb_idx = np.argmax(ucb_values)
        next_candidate = domain[max_ucb_idx].reshape(1, -1)

        # Add the new candidate to the batch
        batch_.append(next_candidate)

        # Append (next_x, predicted_mean(next_x)) to the training dataset
        dist_ = posterior.predict(next_candidate, train_data=dataset)
        hallucinated_next_y = dist_.mean()
        batch_dataset = gpx.Dataset(
            X=np.vstack([batch_dataset.X, next_candidate]),
            y=np.vstack([batch_dataset.y, hallucinated_next_y.reshape(1, 1)]),
        )

        if axes is not None:
            for ax in axes:
                ax.axis("off")

            plot_array(
                axes[b],
                x=np.linspace(-10, 10, 100),
                y=np.linspace(-10, 10, 100),
                array=ucb_values.reshape(100, 100).T,
                vmin=None,
                vmax=None,
            )
            axes[b].scatter(
                next_candidate[0][0],
                next_candidate[0][1],
                color="red",
                s=100,
                marker="x",
            )

            fig = axes[b].get_figure()
            fig.savefig(FIG_DIR / f"gpbucb_{batch_size*iteration + b}.jpg")

    return jnp.vstack(batch_)


def loop(
    num_iterations: int,
    batch_size: int = 6,
    beta: float = 1.0,
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

        batch_to_evaluate = propose_batch(
            posterior=opt_posterior,
            dataset=dataset,
            key=key,
            beta=beta,
            batch_size=batch_size,
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
        axes["sample_1"].set_title(f"GP-BUCB")
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
    fig.savefig(FIG_DIR / f"gpbucb_{batch_size*(iteration+1)}.jpg")


if __name__ == "__main__":
    loop(num_iterations=10, batch_size=6, beta=3.0)
