import torch
import matplotlib.pyplot as plt

from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize

from training import train_model_using_botorch_utils
from initial_design import compute_initial_design_using_sobol
from constants import N_DIMS, DEFAULT_KERNEL
from plotting import (
    plot_predicted_mean,
    plot_predicted_std,
    plot_parity_on_training_data,
    plot_validation_pair_plot,
)

torch.set_default_dtype(torch.float64)


def plot_initial_design(seed: int = 0):
    torch.manual_seed(seed)

    dataset = compute_initial_design_using_sobol(n_points=2 * N_DIMS + 2, n_dimension=2)
    model = SingleTaskGP(
        dataset.X,
        dataset.y,
        covar_module=DEFAULT_KERNEL,
        input_transform=Normalize(N_DIMS),
    )
    model = train_model_using_botorch_utils(model)
    for param, name in model.named_parameters():
        print(f"{name}: {param}")

    fig, (ax_mean, ax_std, ax_parity, ax_loo) = plt.subplots(1, 4, figsize=(4 * 6, 6))
    for ax in [ax_mean, ax_std]:
        ax.scatter(dataset.X[:, 0], dataset.X[:, 1], c=dataset.y)
        ax.set_title("Initial Design")

    plot_predicted_mean(ax_mean, dataset, model)
    plot_predicted_std(ax_std, dataset, model)
    plot_parity_on_training_data(ax_parity, dataset, model)
    plot_validation_pair_plot(ax_loo, dataset)

    plt.show()


if __name__ == "__main__":
    plot_initial_design()
