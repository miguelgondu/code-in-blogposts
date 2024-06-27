import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from poli_baselines.core.abstract_solver import AbstractSolver
from poli_baselines.core.utils.ax.ax_solver import AxSolver

from visualization import selfie_to_numpy_image_array, selfie_to_image

sns.set_theme(style="darkgrid")


def load_alphabet() -> list[str]:
    with open("zinc250k_alphabet_stoi.json", "r") as f:
        alphabet = json.load(f)
    return list(alphabet.keys())


def plot_molecules(
    selfies: list[str], filepath: str, width: int = 200, height: int = 200
):
    fig, axes = plt.subplots(1, len(selfies), figsize=(len(selfies) * 5, 5))

    if len(selfies) == 1:
        axes = [axes]

    for ax, selfie in zip(axes, selfies):
        img = selfie_to_numpy_image_array(selfie, width, height)
        ax.imshow(img)
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(filepath, dpi=300)

    plt.close(fig)


def plot_best_y_curve(best_y: list[np.ndarray], filepath: str):
    best_y = np.array(best_y).flatten()
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.lineplot(x=range(len(best_y)), y=best_y)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best value so far")

    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_experiment(
    solver: AbstractSolver,
    max_iter: int = 100,
    seed: int | None = None,
    experiment_name: str = "random_solver",
):
    solver.solve(max_iter=max_iter, seed=seed)

    # Printing the best performing molecule
    best_value_idx = np.argmax(solver.history["y"])
    best_sequence = "".join(solver.history["x"][best_value_idx].flatten())
    best_value = solver.history["y"][best_value_idx]

    print(f"Best sequence: {best_sequence}")
    print(f"Best value: {best_value}")

    # Plotting the best molecule
    plot_molecules(
        [best_sequence], f"best_molecule_{experiment_name}.jpg", width=1200, height=600
    )

    # Plotting the best_y curve
    plot_best_y_curve(solver.history["best_y"], f"{experiment_name}_best_y_curve.jpg")

    with open(f"{experiment_name}_history.json", "w") as f:
        json.dump(
            {
                "x": [x_i.tolist() for x_i in solver.history["x"]],
                "y": [y_i.tolist() for y_i in solver.history["y"]],
                "best_y": [y_i.tolist() for y_i in solver.history["best_y"]],
            },
            f,
            indent=4,
        )
