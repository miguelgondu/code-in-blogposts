import subprocess

import torch
import numpy as np
import matplotlib.pyplot as plt

from gpytorch.distributions import MultivariateNormal

from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.acquisition import LogExpectedImprovement

import gpjax as gpx
from gpjax.gps import ConjugatePosterior

from batch_bo.functions.objective_function import (
    objective_function,
    compute_objective_function_optima,
)
from batch_bo.dataset import Dataset
from batch_bo.utils.constants import N_DIMS, TOTAL_BUDGET, DEFAULT_KERNEL, LIMITS
from batch_bo.fitting.gp import train_model_using_botorch_utils
from batch_bo.models import ExactGPModel, ExactGPModelJax, ExactGPScikitLearn


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
            for x in torch.linspace(*LIMITS, 100)
            for y in torch.linspace(*LIMITS, 100)
        ]
    )
    z = objective_function(xy_test).reshape(100, 100).T

    ax.contourf(
        np.linspace(*LIMITS, 100),
        np.linspace(*LIMITS, 100),
        z,
        levels=100,
        cmap="viridis",
        vmin=1.0,
        vmax=compute_objective_function_optima(),
    )


def plot_predicted_mean(
    ax: plt.Axes,  # type: ignore
    dataset: Dataset,
    posterior: SingleTaskGP | ExactGPModel | ExactGPModelJax | ExactGPScikitLearn,
):
    if isinstance(posterior, (SingleTaskGP, ExactGPModel)):
        posterior.eval()

    xy_test = torch.Tensor(
        [
            [x, y]
            for x in torch.linspace(*LIMITS, 100)
            for y in torch.linspace(*LIMITS, 100)
        ]
    )

    if isinstance(posterior, (SingleTaskGP, ExactGPModel, ExactGPScikitLearn)):
        min_, max_ = LIMITS
        predictive_dist: MultivariateNormal = posterior.posterior(
            (xy_test - min_) / (max_ - min_)
        )
        predictive_mean = predictive_dist.mean.numpy(force=True)
    elif isinstance(posterior, (ExactGPModelJax, ConjugatePosterior)):
        # GPJax models:
        if isinstance(dataset, gpx.Dataset):
            D = dataset
        elif isinstance(dataset, Dataset):
            D = gpx.Dataset(
                X=dataset.X.numpy(force=True), y=dataset.y.numpy(force=True)
            )
        else:
            raise ValueError("Invalid dataset: ", dataset, type(dataset))
        latent_dist = posterior(xy_test.numpy(force=True), D)
        predictive_dist = posterior.likelihood(latent_dist)
        predictive_mean = predictive_dist.mean()
    else:
        raise ValueError("Invalid posterior: ", posterior, type(posterior))

    ax.contourf(
        np.linspace(*LIMITS, 100),
        np.linspace(*LIMITS, 100),
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
    posterior: SingleTaskGP | ExactGPModel | ExactGPModelJax | ExactGPScikitLearn,
):
    if isinstance(posterior, (SingleTaskGP, ExactGPModel)):
        posterior.eval()

    xy_test = torch.Tensor(
        [
            [x, y]
            for x in torch.linspace(*LIMITS, 100)
            for y in torch.linspace(*LIMITS, 100)
        ]
    )

    if isinstance(posterior, (SingleTaskGP, ExactGPModel, ExactGPScikitLearn)):
        min_, max_ = LIMITS
        predictive_dist: MultivariateNormal = posterior.posterior(
            (xy_test - min_) / (max_ - min_)
        )
        predictive_std = predictive_dist.stddev.numpy(force=True)
    elif isinstance(posterior, (ExactGPModelJax, ConjugatePosterior)):
        # GPJax models:
        if isinstance(dataset, gpx.Dataset):
            D = dataset
        elif isinstance(dataset, Dataset):
            D = gpx.Dataset(
                X=dataset.X.numpy(force=True), y=dataset.y.numpy(force=True)
            )
        else:
            raise ValueError("Invalid dataset: ", dataset, type(dataset))
        latent_dist = posterior(xy_test.numpy(force=True), D)
        predictive_dist = posterior.likelihood(latent_dist)
        predictive_std = predictive_dist.stddev()
    else:
        raise ValueError("Invalid posterior: ", posterior, type(posterior))

    ax.contourf(
        np.linspace(*LIMITS, 100),
        np.linspace(*LIMITS, 100),
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
        [[x, y] for x in np.linspace(*LIMITS, 100) for y in np.linspace(*LIMITS, 100)]
    )
    acq_function = LogExpectedImprovement(posterior, dataset.y.max())

    acq_values = acq_function(xy_test)
    ax.scatter(dataset.X[:, 0], dataset.X[:, 1], c=dataset.y, cmap="viridis")
    ax.contourf(
        np.linspace(*LIMITS, 100),
        np.linspace(*LIMITS, 100),
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


def plot_parity_on_training_data(
    ax: plt.Axes, dataset: Dataset, posterior: SingleTaskGP
):
    actual_values = dataset.y.numpy(force=True).flatten()
    if isinstance(posterior, (SingleTaskGP, ExactGPModel, ExactGPScikitLearn)):
        predictive_dist: MultivariateNormal = posterior.posterior(
            dataset.min_max_scaled_X
        )
        predicted_values = predictive_dist.mean.numpy(force=True)
        error_bars = predictive_dist.stddev.numpy(force=True)
    elif isinstance(posterior, (ExactGPModelJax, ConjugatePosterior)):
        # GPJax models:
        D = gpx.Dataset(X=dataset.X.numpy(force=True), y=dataset.y.numpy(force=True))
        latent_dist = posterior(D.X, D)
        predictive_dist = posterior.likelihood(latent_dist)
        predicted_values = predictive_dist.mean()
        error_bars = predictive_dist.stddev()
    else:
        raise ValueError("Invalid posterior: ", posterior, type(posterior))

    # predicted_values = posterior.posterior(dataset.X).mean.numpy(force=True).flatten()
    # error_bars = posterior.posterior(dataset.X).stddev.numpy(force=True)

    min_ = min(np.concatenate((predicted_values.flatten(), actual_values.flatten())))
    max_ = max(np.concatenate((predicted_values.flatten(), actual_values.flatten())))

    ax.errorbar(
        x=actual_values,
        y=predicted_values.flatten(),
        yerr=error_bars,
        fmt=".k",
        label="mean predictions vs. actual values",
        markersize=10,
        markerfacecolor="white",
        markeredgecolor="gray",
        alpha=0.85,
        # errorbar=test_predictions_95.numpy(force=True),
    )

    padding = 0.05 * (max_ - min_)
    ax.plot(
        np.linspace(min_ - padding, max_ + padding, 100),
        np.linspace(min_ - padding, max_ + padding, 100),
        color="green",
        label="y=x",
        linestyle="--",
        alpha=0.5,
    )
    # ax.set_xlabel("actual values")
    # ax.set_ylabel("mean predictions")
    ax.set_title("Parity plot on training data")
    # ax.legend()
    ax.set_xlim(min_ - padding, max_ + padding)
    ax.set_ylim(min_ - padding, max_ + padding)


def plot_validation_pair_plot(
    ax: plt.Axes,  # type: ignore
    dataset: Dataset,
    percentage: float = 0.25,
):
    X_train = dataset.X
    y_train = dataset.y

    all_predicted_y_values = []
    lower_bounds = []
    upper_bounds = []

    # for index in range(y_train.shape[0]):
    # Split the dataset into training and validation
    indices_for_validation = np.random.choice(
        range(y_train.shape[0]), int(percentage * y_train.shape[0]), replace=False
    )
    other_indices = np.setdiff1d(range(y_train.shape[0]), indices_for_validation)
    X_val = X_train[indices_for_validation]
    y_val = y_train[indices_for_validation]

    X_train_ = X_train[other_indices]
    y_train_ = y_train[other_indices]

    model = SingleTaskGP(
        X_train_,
        y_train_,
        covar_module=DEFAULT_KERNEL,
        input_transform=Normalize(N_DIMS),
    )

    model = train_model_using_botorch_utils(model)

    predictive_dist = model.posterior(X_val)
    predicted_mean = predictive_dist.mean.numpy(force=True).flatten()
    predicted_std = predictive_dist.stddev.numpy(force=True).flatten()

    all_predicted_y_values.append(predicted_mean)
    lower_bounds.append(predicted_mean - predicted_std)
    upper_bounds.append(predicted_mean + predicted_std)

    ax.errorbar(
        x=y_val,
        y=predicted_mean,
        yerr=predicted_std,
        fmt=".k",
        # label="mean predictions vs. actual values",
        markersize=10,
        markerfacecolor="white",
        markeredgecolor="gray",
        alpha=0.85,
    )

    max_predicted = max(np.concatenate(upper_bounds))
    min_predicted = min(np.concatenate(lower_bounds))

    max_ = max(max_predicted, y_train.numpy(force=True).max())
    min_ = min(min_predicted, y_train.numpy(force=True).min())
    padding = 0.05 * (max_ - min_)

    ax.plot(
        np.linspace(min_ - padding, max_ + padding, 100),
        np.linspace(min_ - padding, max_ + padding, 100),
        color="green",
        label="y=x",
        linestyle="--",
        alpha=0.5,
    )
    ax.set_xlim(min_ - padding, max_ + padding)
    ax.set_ylim(min_ - padding, max_ + padding)
    ax.set_title(f"Validation on {int(percentage * 100):3d}% of the data")


def plot_bo_step(posterior: SingleTaskGP, dataset: Dataset, n_iterations: int):
    fig, axes = plt.subplot_mosaic(
        mosaic=[
            [
                "predicted_mean",
                "predicted_mean",
                "predicted_std",
                "predicted_std",
                "parity_plot",
                "parity_plot",
                "validation_parity_plot",
                "validation_parity_plot",
            ],
            [
                "predicted_mean",
                "predicted_mean",
                "predicted_std",
                "predicted_std",
                "parity_plot",
                "parity_plot",
                "validation_parity_plot",
                "validation_parity_plot",
            ],
            [
                "cummulative_regret",
                "cummulative_regret",
                "cummulative_regret",
                "cummulative_regret",
                "cummulative_regret",
                "cummulative_regret",
                "cummulative_regret",
                "cummulative_regret",
            ],
        ],
        height_ratios=[2, 2, 1],
        figsize=(4 * 8, 3 * 4),
    )

    plot_predicted_mean(
        ax=axes["predicted_mean"],
        dataset=dataset,
        posterior=posterior,
    )
    axes["predicted_mean"].axis("off")
    plot_predicted_std(
        ax=axes["predicted_std"],
        dataset=dataset,
        posterior=posterior,
    )
    axes["predicted_std"].axis("off")
    plot_parity_on_training_data(
        ax=axes["parity_plot"],
        dataset=dataset,
        posterior=posterior,
    )
    plot_validation_pair_plot(
        ax=axes["validation_parity_plot"],
        dataset=dataset,
    )
    plot_cummulative_regret(
        ax=axes["cummulative_regret"],
        dataset=dataset,
        total_budget=TOTAL_BUDGET + (2 * N_DIMS + 2),
    )
    return fig


def make_video(
    pattern: str,
    output_filename: str,
    overwrite: bool = True,
    for_quicktime: bool = True,
):
    """This function uses ffmpeg to create a video from a pattern."""

    # ffmpeg -framerate 2 -i gpbucb_%d.jpg -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" gpbucb.mp4
    command = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-framerate",
        "2",
        "-i",
        pattern,
    ]
    if for_quicktime:
        command += [
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
        ]

    command += [
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        output_filename,
    ]
    subprocess.run(command)
