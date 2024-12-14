import numpy as np
import torch
import matplotlib.pyplot as plt

from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize

import gpjax as gpx

from models import ExactGPModel, ExactGPModelJax, ExactGPScikitLearn

from training import (
    train_model_using_botorch_utils,
    train_exact_gp_using_gradient_descent,
    train_exact_gp_jax,
)
from initial_design import compute_initial_design_using_sobol
from constants import N_DIMS, DEFAULT_KERNEL, FIGURES_DIR
from plotting import (
    plot_predicted_mean,
    plot_predicted_std,
    plot_parity_on_training_data,
    plot_validation_pair_plot,
)

torch.set_default_dtype(torch.float64)

FIGURES_DIR = FIGURES_DIR / "initial_design"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)


def plot_initial_design_using_scikit_learn(n_points: int = 10, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    dataset = compute_initial_design_using_sobol(n_points=n_points, n_dimension=2)
    model = ExactGPScikitLearn(
        train_x=dataset.X,
        train_y=dataset.y,
    )

    fig, ((ax_mean, ax_std), (ax_parity, ax_loo)) = plt.subplots(
        2, 2, figsize=(2 * 6, 2 * 6)
    )
    for ax in [ax_mean, ax_std]:
        ax.scatter(dataset.X[:, 0], dataset.X[:, 1], c=dataset.y)

    plot_predicted_mean(ax_mean, dataset, model)
    plot_predicted_std(ax_std, dataset, model)
    plot_parity_on_training_data(ax_parity, dataset, model)
    plot_validation_pair_plot(ax_loo, dataset)

    fig.savefig(
        FIGURES_DIR / f"initial_design_scikit_learn_{n_points:04d}.jpg",
        bbox_inches="tight",
        dpi=300,
    )


def plot_initial_design_using_exact_gps_and_gradient_descent(
    n_points: int = 10, seed: int = 0, gradient_descent_iterations: int = 500
):
    torch.manual_seed(seed)

    dataset = compute_initial_design_using_sobol(n_points=n_points, n_dimension=2)
    model = ExactGPModel(
        train_x=dataset.X,
        train_y=dataset.y.flatten(),
        w_ard=True,
    )

    model = train_exact_gp_using_gradient_descent(
        model, max_nr_iterations=gradient_descent_iterations
    )

    fig, ((ax_mean, ax_std), (ax_parity, ax_loo)) = plt.subplots(
        2, 2, figsize=(2 * 6, 2 * 6)
    )
    for ax in [ax_mean, ax_std]:
        ax.scatter(dataset.X[:, 0], dataset.X[:, 1], c=dataset.y)

    plot_predicted_mean(ax_mean, dataset, model)
    plot_predicted_std(ax_std, dataset, model)
    plot_parity_on_training_data(ax_parity, dataset, model)
    plot_validation_pair_plot(ax_loo, dataset)

    fig.savefig(
        FIGURES_DIR / f"initial_design_gradient_descent_{n_points:04d}.jpg",
        bbox_inches="tight",
        dpi=300,
    )

    # plt.show()


def plot_initial_design_using_botorch(n_points: int = 10, seed: int = 0):
    torch.manual_seed(seed)

    dataset = compute_initial_design_using_sobol(n_points=n_points, n_dimension=2)
    model = SingleTaskGP(
        dataset.X,
        dataset.y,
        covar_module=DEFAULT_KERNEL,
        input_transform=Normalize(N_DIMS),
    )
    model = train_model_using_botorch_utils(model)
    for param, name in model.named_parameters():
        print(f"{name}: {param}")

    fig, ((ax_mean, ax_std), (ax_parity, ax_loo)) = plt.subplots(
        2, 2, figsize=(2 * 6, 2 * 6)
    )
    for ax in [ax_mean, ax_std]:
        ax.scatter(dataset.X[:, 0], dataset.X[:, 1], c=dataset.y)
        ax.set_title("Initial Design")

    plot_predicted_mean(ax_mean, dataset, model)
    plot_predicted_std(ax_std, dataset, model)
    plot_parity_on_training_data(ax_parity, dataset, model)
    plot_validation_pair_plot(ax_loo, dataset)

    fig.savefig(
        FIGURES_DIR / f"initial_design_botorch_{n_points:04d}.jpg",
        bbox_inches="tight",
        dpi=300,
    )

    # plt.show()


def plot_initial_design_using_jax_models(
    n_points: int = 10, seed: int = 0, max_nr_iterations: int = 500
):
    dataset = compute_initial_design_using_sobol(n_points=n_points, n_dimension=2)
    model = ExactGPModelJax(
        train_x=dataset.X,
        train_y=dataset.y,
    )

    model = train_exact_gp_jax(model, max_nr_iterations=max_nr_iterations, seed=seed)

    fig, ((ax_mean, ax_std), (ax_parity, ax_loo)) = plt.subplots(
        2, 2, figsize=(2 * 6, 2 * 6)
    )
    for ax in [ax_mean, ax_std]:
        ax.scatter(dataset.X[:, 0], dataset.X[:, 1], c=dataset.y)
        ax.set_title("Initial Design")

    plot_predicted_mean(ax_mean, dataset, model)
    plot_predicted_std(ax_std, dataset, model)
    plot_parity_on_training_data(ax_parity, dataset, model)
    plot_validation_pair_plot(ax_loo, dataset)

    fig.savefig(
        FIGURES_DIR / f"initial_design_gpjax_{n_points:04d}.jpg",
        bbox_inches="tight",
        dpi=300,
    )

    # plt.show()


if __name__ == "__main__":
    for n_points in range(5, 25, 1):
        # plot_initial_design_using_botorch(n_points=n_points)
        # plot_initial_design_using_exact_gps_and_gradient_descent(
        #     n_points=n_points,
        #     gradient_descent_iterations=n_points * 100,
        # )
        # plot_initial_design_using_jax_models(
        #     n_points=n_points,
        #     max_nr_iterations=n_points * 100,
        # )
        plot_initial_design_using_scikit_learn(n_points=n_points)
        plt.close("all")
