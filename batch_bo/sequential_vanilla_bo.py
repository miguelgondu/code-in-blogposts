import matplotlib.pyplot as plt
import torch
import numpy as np
from botorch.models import SingleTaskGP
import gpytorch
from botorch.models.transforms.input import Normalize

from objective_function import compute_domain, objective_function, N_DIMS
from initial_design import compute_initial_design_using_sobol
from training import train_model_using_botorch_utils
from plotting import plot_predicted_mean, plot_cummulative_regret
from dataset import Dataset
from constants import FIGURES_DIR

torch.set_default_dtype(torch.float64)

DEFAULT_KERNEL = gpytorch.kernels.ScaleKernel(
    gpytorch.kernels.RBFKernel(
        ard_num_dims=N_DIMS,
        lengthscale_prior=gpytorch.priors.LogNormalPrior(np.log(N_DIMS) / 2, 1.0),
    )
)


def run_sequential_vanilla_bo_using_thompson_sampling(n_iterations: int, seed: int = 0):
    torch.manual_seed(seed)

    dataset = compute_initial_design_using_sobol(n_points=2 * N_DIMS + 2, n_dimension=2)

    for iteration in range(n_iterations):
        model = SingleTaskGP(
            dataset.X,
            dataset.y,
            covar_module=DEFAULT_KERNEL,
            input_transform=Normalize(N_DIMS),
        )
        model = train_model_using_botorch_utils(model)

        fig = plot_bo_step(model, dataset, n_iterations=n_iterations)
        fig.savefig(
            FIGURES_DIR / f"sequential_vanilla_bo_{iteration:09d}.png",
            bbox_inches="tight",
        )
        plt.close(fig)

        domain = compute_domain()
        dist_ = model(domain)
        one_sample = dist_.sample()

        x_next = domain[one_sample.argmax()].unsqueeze(0)
        y_next = objective_function(x_next)

        dataset = Dataset(
            X=torch.cat([dataset.X, x_next]), y=torch.cat([dataset.y, y_next])
        )
        print(f"New data point: {x_next} - Value: {y_next}")

    fig = plot_bo_step(model, dataset, n_iterations)
    fig.savefig(
        FIGURES_DIR / f"sequential_vanilla_bo_{iteration+1:09d}.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    plt.show()


def plot_bo_step(posterior: SingleTaskGP, dataset: Dataset, n_iterations: int):
    fig, axes = plt.subplot_mosaic(
        mosaic=[
            ["predicted_mean", "predicted_mean"],
            ["predicted_mean", "predicted_mean"],
            ["cummulative_regret", "cummulative_regret"],
        ],
        height_ratios=[2, 2, 2],
        figsize=(5 * 5, 5 * 3),
    )

    plot_predicted_mean(
        ax=axes["predicted_mean"],
        dataset=dataset,
        posterior=posterior,
    )
    axes["predicted_mean"].axis("off")
    plot_cummulative_regret(
        ax=axes["cummulative_regret"],
        dataset=dataset,
        total_budget=100,
    )
    return fig


if __name__ == "__main__":
    run_sequential_vanilla_bo_using_thompson_sampling(100)
