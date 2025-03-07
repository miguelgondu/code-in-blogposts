from pathlib import Path
from jax import config

from poli.repository import ToyContinuousBlackBox

import numpy as np
from scipy.stats.qmc import Sobol

import gpjax as gpx
import jax.numpy as jnp
import jax.random as jr

import seaborn as sns
import matplotlib.pyplot as plt

from batch_bo.plotting import (
    plot_array,
    plot_predicted_mean,
    plot_cummulative_regret,
)
from batch_bo.fitting.gp import fit_gp
from batch_bo.utils.constants import ROOT_DIR, LIMITS

config.update("jax_enable_x64", True)

FIG_DIR = ROOT_DIR / "figures" / "batch_ts"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def sample_gp(
    domain: np.ndarray,
    posterior,
    dataset: gpx.Dataset,
    key: jr.PRNGKey,
    num_samples: int = 1,
) -> jnp.ndarray:
    latent_dist = posterior(domain, dataset)
    # predictive_dist = posterior.likelihood(latent_dist)

    return latent_dist.sample(seed=key, sample_shape=(num_samples,))


def batch_ts(num_iterations: int, batch_size: int = 6):
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

    # scaling the samples
    min_, max_ = LIMITS
    samples = samples * (max_ - min_) + min_
    noisy_evaluations = f(samples)  # + 0.25 * jr.normal(subkey, shape=(10, 1))

    dataset = gpx.Dataset(X=samples, y=noisy_evaluations)

    domain = np.array(
        [
            [x, y]
            for x in np.linspace(min_, max_, 100)
            for y in np.linspace(min_, max_, 100)
        ]
    )

    for iteration in range(num_iterations):
        key = jr.PRNGKey(iteration)

        # Fit the GP with the dataset we have so far
        # print(dataset)
        # print(dataset.X.shape)
        opt_posterior = fit_gp(dataset, key)

        # Sample it batch_size times
        samples = sample_gp(
            domain=domain,
            posterior=opt_posterior,
            dataset=dataset,
            key=key,
            num_samples=batch_size,
        )

        # Optimize each and every sample
        batch_to_evaluate = domain[samples.argmax(axis=1)]

        # Visualize this round
        plot_batch_ts(
            dataset=dataset,
            domain=domain,
            opt_posterior=opt_posterior,
            samples=samples,
            batch_to_evaluate=batch_to_evaluate,
            filename=FIG_DIR / f"batch_ts_{iteration}.jpg",
            total_budget=10 + batch_size * num_iterations + 1,
        )

        # Evaluate the function at the new batch
        evaluations = f(batch_to_evaluate)
        noisy_evaluations = (
            evaluations  #  + 0.25 * jr.normal(key, shape=(batch_size, 1))
        )

        # Add the new evaluations to the dataset
        new_dataset = gpx.Dataset(X=batch_to_evaluate, y=noisy_evaluations)
        dataset += new_dataset

    # Final round visualization
    plot_batch_ts(
        dataset=dataset,
        domain=domain,
        opt_posterior=opt_posterior,
        samples=samples,
        batch_to_evaluate=batch_to_evaluate,
        filename=FIG_DIR / f"batch_ts_{iteration+1}.jpg",
        total_budget=10 + batch_size * num_iterations + 1,
    )


def plot_batch_ts(
    dataset: gpx.Dataset,
    domain: np.ndarray,
    opt_posterior,
    samples: np.ndarray,
    batch_to_evaluate: np.ndarray,
    filename: str = None,
    total_budget: int = 100,
):
    f = ToyContinuousBlackBox("cross_in_tray", 2)
    # Let's start by making a mosaic of the samples
    # focusing on the current predicted mean.
    fig, axes = plt.subplot_mosaic(
        mosaic=[
            ["predicted_mean", "predicted_mean", "sample_0", "sample_1", "sample_2"],
            ["predicted_mean", "predicted_mean", "sample_3", "sample_4", "sample_5"],
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

    # Predictive mean
    plot_predicted_mean(
        axes["predicted_mean"],
        dataset=dataset,
        posterior=opt_posterior,
    )
    axes["predicted_mean"].axis("off")

    # Samples
    for i in range(6):
        ax = axes[f"sample_{i}"]
        ax.axis("off")
        plot_array(
            ax=ax,
            x=np.linspace(*LIMITS, 100),
            y=np.linspace(*LIMITS, 100),
            array=samples[i].reshape(100, 100).T,
            vmin=1.0,
            vmax=f.function.optima,
        )
        ax.scatter(
            batch_to_evaluate[i][0],
            batch_to_evaluate[i][1],
            color="red",
            s=100,
            marker="x",
        )

    plot_cummulative_regret(
        axes["cummulative_regret"],
        dataset,
        total_budget=total_budget,
    )

    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    batch_ts(10)
