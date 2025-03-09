import numpy as np
import matplotlib.pyplot as plt
from typing import Type

import seaborn as sns
import torch

from poli.repository import ToyContinuousBlackBox

from batch_bo.initial_design.sobol import compute_initial_design_using_sobol
from batch_bo.utils.constants import (
    LIMITS,
    RESOLUTION,
    FIGURES_DIR,
    N_DIMS,
    SEED,
    FUNCTION_NAME,
)
from batch_bo.models.gp import ExactGPScikitLearn, ExactGPModelJax, ExactGPModel
from batch_bo.functions.objective_function import compute_domain
from batch_bo.plotting import plot_parity_on_training_data

N_POINTS = 10


def study_perturbations_to_y(
    GPClass: Type[ExactGPScikitLearn | ExactGPModelJax | ExactGPModel],
):
    sns.set_theme(style="darkgrid", font_scale=2.0)
    # f = ToyContinuousBlackBox(function_name=FUNCTION_NAME, n_dimensions=N_DIMS)

    index_to_perturb = 8

    # Sobol sampling
    for i, perturbation in enumerate(np.linspace(-1.0, 1.0, 40)):
        dataset = compute_initial_design_using_sobol(
            n_points=N_POINTS, n_dimension=N_DIMS, seed=SEED
        )
        domain = compute_domain()
        bespoke_y = torch.Tensor(
            [
                [1.0000],
                [2.2944],
                [2.2186],
                [2.2186],
                [2.1865],
                [2.4786],
                [2.0503],
                [2.0503],
                [2.0189],
                [2.4318],
            ]
        )
        bespoke_y[index_to_perturb] += perturbation
        dataset.y = bespoke_y
        gp = GPClass(train_x=dataset.X, train_y=bespoke_y)
        dist_ = gp.posterior(domain)

        fig, axes = plt.subplots(1, 3, figsize=(6 * 3, 6))
        axes[0].contourf(
            np.linspace(*LIMITS, RESOLUTION),
            np.linspace(*LIMITS, RESOLUTION),
            dist_.mean.reshape(RESOLUTION, RESOLUTION).T,
            levels=100,
            cmap="viridis",
            vmin=min(bespoke_y),
            vmax=max(bespoke_y),
        )
        axes[0].scatter(dataset.X[:, 0], dataset.X[:, 1], c="k", marker="o")
        axes[0].scatter(
            dataset.X[index_to_perturb, 0],
            dataset.X[index_to_perturb, 1],
            c="r",
            marker="x",
        )
        axes[0].axis("off")
        axes[0].set_title("Predictive mean")
        axes[1].contourf(
            np.linspace(*LIMITS, RESOLUTION),
            np.linspace(*LIMITS, RESOLUTION),
            dist_.scale.reshape(RESOLUTION, RESOLUTION).T,
            levels=100,
            cmap="viridis",
        )
        axes[1].scatter(dataset.X[:, 0], dataset.X[:, 1], c="k", marker="o")
        axes[1].scatter(
            dataset.X[index_to_perturb, 0],
            dataset.X[index_to_perturb, 1],
            c="r",
            marker="x",
        )
        axes[1].axis("off")
        axes[1].set_title("Predictive std")
        plot_parity_on_training_data(axes[2], dataset, gp)

        fig.tight_layout()
        PERTURBATION_FIGURES_DIR = FIGURES_DIR / "perturbations"
        PERTURBATION_FIGURES_DIR.mkdir(exist_ok=True, parents=True)
        fig.savefig(
            PERTURBATION_FIGURES_DIR
            / f"perturbation_{str(GPClass.__name__)}_{i:02d}.jpg",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
        # plt.show()


if __name__ == "__main__":
    # fit_gp_to_sobol_samples(ExactGPScikitLearn)
    study_perturbations_to_y(ExactGPModelJax)
    # fit_gp_to_sobol_samples(ExactGPModel)
