from pathlib import Path
import sys

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

from expected_improvement import ExpectedImprovement

THIS_DIR = Path(__file__).resolve().parent
FIG_DIR = THIS_DIR / "figures" / "sequential_bo"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# seaborn.set_theme(style="darkgrid", font_scale=2.0)


def plot_guiding_example():
    function_name = "cross_in_tray"
    n_dims = 2
    f = ToyContinuousBlackBox(function_name=function_name, n_dimensions=n_dims)

    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    xy = np.array(np.meshgrid(x, y)).T.reshape(-1, n_dims)

    z = f(xy)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection="3d")

    # Adding the optima
    print(f.function.optima_location)
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
        THIS_DIR / f"{function_name}_{n_dims}d.jpg", dpi=300, bbox_inches="tight"
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
        [[x, y] for x in np.linspace(-10, 10, 100) for y in np.linspace(-10, 10, 100)]
    )

    latent_dist = opt_posterior(xy_test, dataset)
    predictive_dist = opt_posterior.likelihood(latent_dist)
    predictive_mean = predictive_dist.mean()
    predictive_std = predictive_dist.stddev()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].contourf(
        np.linspace(-10, 10, 100),
        np.linspace(-10, 10, 100),
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
        np.linspace(-10, 10, 100),
        np.linspace(-10, 10, 100),
        predictive_std.reshape(100, 100).T,
        levels=100,
        cmap="viridis",
    )
    axes[1].scatter(samples[:, 0], samples[:, 1], c="k", marker="o")
    axes[1].axis("off")
    axes[1].set_title("Predictive std")

    fig.tight_layout()
    fig.savefig(
        THIS_DIR / f"{function_name}_{n_dims}d_gp.jpg", dpi=300, bbox_inches="tight"
    )

    plt.show()


def fit_gp(dataset: gpx.Dataset, key: jr.PRNGKey) -> gpx.Module:
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

    return opt_posterior


def bo_loop(dataset: gpx.Dataset, key: jr.PRNGKey) -> jnp.ndarray:
    negative_dataset = gpx.Dataset(X=dataset.X, y=-dataset.y)
    opt_posterior = fit_gp(negative_dataset, key)

    # Define the utility function
    # ei_factory = ExpectedImprovement()
    ei_factory = ProbabilityOfImprovement()
    ei_fn = ei_factory.build_utility_function(
        posteriors={"OBJECTIVE": opt_posterior},
        datasets={"OBJECTIVE": negative_dataset},
        key=key,
    )

    xy_test = np.array(
        [[x, y] for x in np.linspace(-10, 10, 100) for y in np.linspace(-10, 10, 100)]
    )
    acq_vals = ei_fn(xy_test)

    next_candidate = xy_test[jnp.argmax(acq_vals)]

    return next_candidate


def plot_array(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    array: np.ndarray,
    vmin: float,
    vmax: float,
):
    ax.contourf(x, y, array, levels=100, cmap="viridis", vmin=vmin, vmax=vmax)


def plot_predicted_mean(
    ax: plt.Axes, dataset: gpx.Dataset, posterior: gpx.Module, f: ToyContinuousBlackBox
):
    xy_test = np.array(
        [[x, y] for x in np.linspace(-10, 10, 100) for y in np.linspace(-10, 10, 100)]
    )

    latent_dist = posterior(xy_test, dataset)
    predictive_dist = posterior.likelihood(latent_dist)
    predictive_mean = predictive_dist.mean()

    ax.contourf(
        np.linspace(-10, 10, 100),
        np.linspace(-10, 10, 100),
        predictive_mean.reshape(100, 100).T,
        levels=100,
        cmap="viridis",
        vmin=1.0,
        vmax=f.function.optima,
    )
    ax.scatter(dataset.X[:, 0], dataset.X[:, 1], c="k", marker="o")
    ax.axis("off")
    ax.set_title("Predictive mean")


def plot_predicted_std(
    ax: plt.Axes, dataset: gpx.Dataset, posterior: gpx.Module, f: ToyContinuousBlackBox
):
    xy_test = np.array(
        [[x, y] for x in np.linspace(-10, 10, 100) for y in np.linspace(-10, 10, 100)]
    )

    latent_dist = posterior(xy_test, dataset)
    predictive_dist = posterior.likelihood(latent_dist)
    predictive_std = predictive_dist.stddev()
    ax.scatter(dataset.X[:, 0], dataset.X[:, 1], c=dataset.y, cmap="viridis")
    ax.contourf(
        np.linspace(-10, 10, 100),
        np.linspace(-10, 10, 100),
        predictive_std.reshape(100, 100).T,
        levels=100,
        cmap="viridis",
    )
    ax.scatter(dataset.X[:, 0], dataset.X[:, 1], c="k", marker="o")
    ax.axis("off")
    ax.set_title("Predictive std")


def plot_acq_function(
    ax: plt.Axes,
    dataset: gpx.Dataset,
    posterior: gpx.Module,
):
    xy_test = np.array(
        [[x, y] for x in np.linspace(-10, 10, 100) for y in np.linspace(-10, 10, 100)]
    )
    acq_function = ExpectedImprovement().build_utility_function(
        posteriors={"OBJECTIVE": posterior},
        datasets={"OBJECTIVE": gpx.Dataset(X=dataset.X, y=-dataset.y)},
        key=jr.PRNGKey(0),
    )

    acq_values = acq_function(xy_test)
    ax.scatter(dataset.X[:, 0], dataset.X[:, 1], c=dataset.y, cmap="viridis")
    ax.contourf(
        np.linspace(-10, 10, 100),
        np.linspace(-10, 10, 100),
        acq_values.reshape(100, 100).T,
        levels=100,
        cmap="viridis",
    )
    ax.scatter(dataset.X[:, 0], dataset.X[:, 1], c="k", marker="o")
    ax.axis("off")
    ax.set_title("Acq. function")


def plot_cummulative_regret(
    ax: plt.Axes,
    dataset: gpx.Dataset,
    f: ToyContinuousBlackBox,
    total_budget: int = 100,
    log_scale: bool = True,
):
    best_so_far = np.maximum.accumulate(dataset.y)
    regret = np.abs(f.function.optima - best_so_far)
    ax.plot(regret)
    # ax.axis("off")
    ax.set_xlim(0, total_budget)
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylim(10**-4, 10**0)
    ax.set_title("|real optimum - best so far|")


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
    samples = samples * (10 - (-10)) - 10
    noisy_evaluations = f(samples)  #  + 0.25 * jr.normal(subkey, shape=(10, 1))

    dataset = gpx.Dataset(X=samples, y=noisy_evaluations)

    for i in range(50):
        key = jr.PRNGKey(i)
        next_candidate = bo_loop(dataset, key).reshape(1, -1)
        next_evaluation = f(next_candidate)  # + 0.25 * jr.normal(key, shape=(1, 1))
        dataset = gpx.Dataset(
            X=np.vstack([dataset.X, next_candidate]),
            y=np.vstack([dataset.y, next_evaluation]),
        )

        posterior = fit_gp(dataset, key)

        # visualizing
        fig, axes = plt.subplots(1, 4, figsize=(5 * 4, 5))
        plot_predicted_mean(axes[0], dataset, posterior, f)
        plot_predicted_std(axes[1], dataset, posterior, f)
        plot_acq_function(axes[2], dataset, posterior)
        plot_cummulative_regret(axes[3], dataset, f, total_budget=60)

        fig.tight_layout()
        fig.savefig(
            FIG_DIR / f"{function_name}_{n_dims}d_gp_PI_{i}.jpg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


if __name__ == "__main__":
    # plot_guiding_example()
    # fit_a_gp_to_sobol_samples()
    bo()
