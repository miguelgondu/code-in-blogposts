import torch
import numpy as np
import matplotlib.pyplot as plt

from gpytorch.distributions import MultivariateNormal

from botorch.models import SingleTaskGP
from botorch.acquisition import LogExpectedImprovement

from objective_function import objective_function, compute_objective_function_optima
from dataset import Dataset


def plot_array(
    ax: plt.Axes,  # type: ignore
    x: np.ndarray,
    y: np.ndarray,
    array: np.ndarray,
    vmin: float,
    vmax: float,
):
    ax.contourf(x, y, array, levels=100, cmap="viridis", vmin=vmin, vmax=vmax)


def plot_objective_function(ax: plt.Axes):
    xy_test = torch.Tensor(
        [
            [x, y]
            for x in torch.linspace(-10, 10, 100)
            for y in torch.linspace(-10, 10, 100)
        ]
    )
    z = objective_function(xy_test).reshape(100, 100).T

    ax.contourf(
        np.linspace(-10, 10, 100),
        np.linspace(-10, 10, 100),
        z,
        levels=100,
        cmap="viridis",
        vmin=1.0,
        vmax=compute_objective_function_optima(),
    )


def plot_predicted_mean(
    ax: plt.Axes,  # type: ignore
    dataset: Dataset,
    posterior: SingleTaskGP,
):
    posterior.eval()
    xy_test = torch.Tensor(
        [
            [x, y]
            for x in torch.linspace(-10, 10, 100)
            for y in torch.linspace(-10, 10, 100)
        ]
    )

    predictive_dist: MultivariateNormal = posterior.posterior(xy_test)
    predictive_mean = predictive_dist.mean.numpy(force=True)

    ax.contourf(
        np.linspace(-10, 10, 100),
        np.linspace(-10, 10, 100),
        predictive_mean.reshape(100, 100).T,
        levels=100,
        cmap="viridis",
        vmin=1.0,
        vmax=compute_objective_function_optima(),
    )
    ax.scatter(dataset.X[:, 0], dataset.X[:, 1], c="k", marker="o")
    ax.axis("off")
    ax.set_title("Predictive mean")


def plot_predicted_std(
    ax: plt.Axes,  # type: ignore
    dataset: Dataset,
    posterior: SingleTaskGP,
):
    posterior.eval()
    xy_test = np.array(
        [[x, y] for x in np.linspace(-10, 10, 100) for y in np.linspace(-10, 10, 100)]
    )

    predictive_dist: MultivariateNormal = posterior.posterior(xy_test)
    predictive_std = predictive_dist.stddev
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
    ax: plt.Axes,  # type: ignore
    dataset: Dataset,
    posterior: SingleTaskGP,
):
    posterior.eval()
    xy_test = np.array(
        [[x, y] for x in np.linspace(-10, 10, 100) for y in np.linspace(-10, 10, 100)]
    )
    acq_function = LogExpectedImprovement(posterior, dataset.y.max())

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
    ax: plt.Axes,  # type: ignore
    dataset: Dataset,
    total_budget: int = 100,
    log_scale: bool = True,
):
    best_so_far = np.maximum.accumulate(dataset.y.flatten())
    regret = np.abs(compute_objective_function_optima() - best_so_far)
    ax.plot(regret)
    ax.set_xlim(0, total_budget)
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylim(10**-4, 10**0)
    # ax.set_title("|real optimum - best so far|")
