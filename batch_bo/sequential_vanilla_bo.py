import matplotlib.pyplot as plt
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize

from constants import FIGURES_DIR, DEFAULT_KERNEL, N_DIMS, SEED, TOTAL_BUDGET
from objective_function import compute_domain, objective_function
from initial_design import compute_initial_design_using_sobol
from training import train_model_using_botorch_utils
from plotting import (
    plot_bo_step,
)
from dataset import Dataset

torch.set_default_dtype(torch.float64)


def run_sequential_vanilla_bo_using_thompson_sampling():
    torch.manual_seed(SEED)

    dataset = compute_initial_design_using_sobol(n_points=2 * N_DIMS + 2, n_dimension=2)

    for iteration in range(TOTAL_BUDGET):
        model = SingleTaskGP(
            dataset.X,
            dataset.y,
            covar_module=DEFAULT_KERNEL,
            input_transform=Normalize(N_DIMS),
        )
        model = train_model_using_botorch_utils(model)

        fig = plot_bo_step(model, dataset, n_iterations=TOTAL_BUDGET)
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
        print(
            f"New data point: {x_next} - Value: {y_next} - Best so far: {dataset.y.max()}"
        )

    fig = plot_bo_step(model, dataset, TOTAL_BUDGET)
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / f"sequential_vanilla_bo_{iteration+1:09d}.png",
        bbox_inches="tight",
    )
    plt.close(fig)

    plt.show()


if __name__ == "__main__":
    run_sequential_vanilla_bo_using_thompson_sampling()
