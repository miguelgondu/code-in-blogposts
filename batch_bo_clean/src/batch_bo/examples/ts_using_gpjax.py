from jax import config


from poli.repository import ToyContinuousBlackBox

import numpy as np
from scipy.stats.qmc import Sobol

import gpjax as gpx
from jax import jit
import jax.numpy as jnp
import jax.random as jr
import optax as ox


import seaborn as sns
import matplotlib.pyplot as plt

from batch_bo.plotting.plotting import (
    plot_predicted_mean,
    plot_predicted_std,
    plot_array,
    plot_cummulative_regret,
)
from batch_bo.fitting.gp import fit_gp
from batch_bo.utils.constants import ROOT_DIR, LIMITS

config.update("jax_enable_x64", True)

FIG_DIR = ROOT_DIR / "figures" / "sequential_bo"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_guiding_example():
    function_name = "cross_in_tray"
    n_dims = 2
    f = ToyContinuousBlackBox(function_name=function_name, n_dimensions=n_dims)

    x = np.linspace(*LIMITS, 100)
    y = np.linspace(*LIMITS, 100)
    xy = np.array(np.meshgrid(x, y)).T.reshape(-1, n_dims)

    z = f(xy)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection="3d")

    # Adding the optima
    # print(f.function.optima_location)
    optimum = f.function.optima_location
    ax.scatter(
        optimum[0],
        optimum[1],
        f(optimum.reshape(1, -1)),
        c="k",
        s=150,
        marker="x",
    )

    ax.plot_trisurf(xy[:, 0], xy[:, 1], z.flatten(), cmap="viridis")

    ax2 = fig.add_subplot(122)
    ax2.contourf(x, y, z.reshape(100, 100), levels=100, cmap="viridis")
    ax2.scatter(
        optimum[0],
        optimum[1],
        c="k",
        s=150,
        marker="x",
    )

    fig.tight_layout()
    fig.savefig(
        FIG_DIR / f"{function_name}_{n_dims}d.jpg", dpi=300, bbox_inches="tight"
    )

    plt.show()


def fit_a_gp_to_sobol_samples():
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

    # Construct the prior
    meanf = gpx.mean_functions.Zero()
    kernel = gpx.kernels.RBF()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

    # Define a likelihood
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=10)

    # Construct the posterior
    posterior = prior * likelihood

    # Define an optimiser
    optimiser = ox.adam(learning_rate=1e-2)

    # Define the marginal log-likelihood
    negative_mll = jit(gpx.objectives.ConjugateMLL(negative=True))

    # Obtain Type 2 MLEs of the hyperparameters
    opt_posterior, history = gpx.fit(
        model=posterior,
        objective=negative_mll,
        train_data=dataset,
        optim=optimiser,
        num_iters=500,
        safe=True,
        key=key,
    )

    xy_test = np.array(
        [[x, y] for x in np.linspace(*LIMITS, 100) for y in np.linspace(*LIMITS, 100)]
    )

    latent_dist = opt_posterior(xy_test, dataset)
    predictive_dist = opt_posterior.likelihood(latent_dist)
    predictive_mean = predictive_dist.mean()
    predictive_std = predictive_dist.stddev()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].contourf(
        np.linspace(*LIMITS, 100),
        np.linspace(*LIMITS, 100),
        predictive_mean.reshape(100, 100).T,
        levels=100,
        cmap="viridis",
        vmin=1.0,
        vmax=f.function.optima,
    )
    axes[0].scatter(samples[:, 0], samples[:, 1], c="k", marker="o")
    axes[0].axis("off")
    axes[0].set_title("Predictive mean")
    axes[1].scatter(samples[:, 0], samples[:, 1], c=noisy_evaluations, cmap="viridis")
    axes[1].contourf(
        np.linspace(*LIMITS, 100),
        np.linspace(*LIMITS, 100),
        predictive_std.reshape(100, 100).T,
        levels=100,
        cmap="viridis",
    )
    axes[1].scatter(samples[:, 0], samples[:, 1], c="k", marker="o")
    axes[1].axis("off")
    axes[1].set_title("Predictive std")

    fig.tight_layout()
    fig.savefig(
        FIG_DIR / f"{function_name}_{n_dims}d_gp.jpg", dpi=300, bbox_inches="tight"
    )

    plt.show()


def bo_loop(dataset: gpx.Dataset, key: jr.PRNGKey) -> jnp.ndarray:
    # negative_dataset = gpx.Dataset(X=dataset.X, y=-dataset.y)
    opt_posterior = fit_gp(dataset, key)

    # Define the utility function
    xy_test = np.array(
        [[x, y] for x in np.linspace(*LIMITS, 100) for y in np.linspace(*LIMITS, 100)]
    )
    ts_sample = opt_posterior.predict(xy_test, train_data=dataset).sample(
        key, sample_shape=(1,)
    )
    next_candidate = xy_test[jnp.argmax(ts_sample)]

    return next_candidate, ts_sample


def bo():
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
    min_, max_ = LIMITS
    samples = samples * (max_ - min_) + min_
    noisy_evaluations = f(samples)  #  + 0.25 * jr.normal(subkey, shape=(10, 1))

    dataset = gpx.Dataset(X=samples, y=noisy_evaluations)

    for i in range(50):
        key = jr.PRNGKey(i)
        next_candidate, ts_sample = bo_loop(dataset, key)
        next_candidate = next_candidate.reshape(1, -1)
        next_evaluation = f(next_candidate)  # + 0.25 * jr.normal(key, shape=(1, 1))
        dataset = gpx.Dataset(
            X=np.vstack([dataset.X, next_candidate]),
            y=np.vstack([dataset.y, next_evaluation]),
        )

        posterior = fit_gp(dataset, key)

        # visualizing
        fig, axes = plt.subplot_mosaic(
            mosaic=[
                ["mean", "std", "acq"],
                ["regret", "regret", "regret"],
            ],
            height_ratios=[4, 1.5],
            figsize=(5 * 5, 5 * 2.05),
        )
        plot_predicted_mean(axes["mean"], dataset, posterior)
        plot_predicted_std(axes["std"], dataset, posterior)
        plot_array(
            axes["acq"],
            np.linspace(*LIMITS, 100),
            np.linspace(*LIMITS, 100),
            ts_sample.reshape(100, 100).T,
            vmin=ts_sample.min(),
            vmax=ts_sample.max(),
        )
        axes["acq"].scatter(
            next_candidate[0, 0], next_candidate[0, 1], c="r", marker="x"
        )
        plot_cummulative_regret(axes["regret"], dataset, total_budget=60)

        fig.tight_layout()
        fig.savefig(
            FIG_DIR / f"{function_name}_{n_dims}d_gp_TS_{i}.jpg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


if __name__ == "__main__":
    # plot_guiding_example()
    # fit_a_gp_to_sobol_samples()
    bo()
