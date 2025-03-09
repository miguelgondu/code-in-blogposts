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
    INITIAL_DESIGN_SIZE,
)
from batch_bo.models.gp import ExactGPScikitLearn, ExactGPModelJax, ExactGPModel
from batch_bo.functions.objective_function import compute_domain
from batch_bo.plotting import plot_parity_on_training_data

torch.set_default_dtype(torch.float64)


def fit_gp_to_sobol_samples(
    GPClass: Type[ExactGPScikitLearn | ExactGPModelJax | ExactGPModel],
):
    sns.set_theme(style="darkgrid", font_scale=2.0)
    f = ToyContinuousBlackBox(function_name=FUNCTION_NAME, n_dimensions=N_DIMS)

    # Sobol sampling
    dataset = compute_initial_design_using_sobol(
        n_points=INITIAL_DESIGN_SIZE, n_dimension=N_DIMS, seed=SEED
    )
    domain = compute_domain()
    gp = GPClass(train_x=dataset.X, train_y=dataset.y)
    dist_ = gp.posterior(domain)

    fig, axes = plt.subplots(1, 3, figsize=(6 * 3, 6))
    axes[0].contourf(
        np.linspace(*LIMITS, RESOLUTION),
        np.linspace(*LIMITS, RESOLUTION),
        dist_.mean.reshape(RESOLUTION, RESOLUTION).T,
        levels=100,
        cmap="viridis",
        # vmin=1.0,
        # vmax=f.function.optima,
    )
    axes[0].scatter(dataset.X[:, 0], dataset.X[:, 1], c="k", marker="o")
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
    axes[1].axis("off")
    axes[1].set_title("Predictive std")
    plot_parity_on_training_data(axes[2], dataset, gp)

    fig.tight_layout()
    INITIAL_DESIGN_FIGURES_DIR = FIGURES_DIR / "initial_design"
    INITIAL_DESIGN_FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    fig.savefig(
        INITIAL_DESIGN_FIGURES_DIR / f"initial_fit_{str(GPClass.__name__)}.jpg",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


if __name__ == "__main__":
    fit_gp_to_sobol_samples(ExactGPScikitLearn)
    fit_gp_to_sobol_samples(ExactGPModelJax)
    fit_gp_to_sobol_samples(ExactGPModel)
