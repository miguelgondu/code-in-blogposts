import numpy as np
import matplotlib.pyplot as plt

from poli.repository import ToyContinuousBlackBox
import torch

from batch_bo.utils.constants import (
    FIGURES_DIR,
    FUNCTION_NAME,
    N_DIMS,
    LIMITS,
    RESOLUTION,
)
from batch_bo.plotting import plot_objective_function
from batch_bo.functions.objective_function import objective_function

if __name__ == "__main__":
    f = ToyContinuousBlackBox(function_name=FUNCTION_NAME, n_dimensions=N_DIMS)
    x = np.linspace(*LIMITS, RESOLUTION)
    y = np.linspace(*LIMITS, RESOLUTION)
    xy = np.array(np.meshgrid(x, y)).T.reshape(-1, N_DIMS)

    z = objective_function(torch.from_numpy(xy))

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(121, projection="3d")

    # Adding the optima
    # print(f.function.optima_location)
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
    plot_objective_function(ax2)
    ax2.scatter(
        optimum[0],
        optimum[1],
        c="k",
        s=150,
        marker="x",
    )

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / f"{FUNCTION_NAME}_{N_DIMS}d.jpg", dpi=300, bbox_inches="tight"
    )
    plt.show()
    plt.close()
