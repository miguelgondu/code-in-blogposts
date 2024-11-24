import torch
import numpy as np

from poli.repository import ToyContinuousBlackBox

FUNCTION_NAME = "cross_in_tray"
N_DIMS = 2


def compute_domain() -> torch.Tensor:
    n_dims = 2
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    xy = np.array(np.meshgrid(x, y)).T.reshape(-1, n_dims)
    return torch.from_numpy(xy).to(torch.get_default_dtype())


def objective_function(x: torch.Tensor) -> torch.Tensor:
    f = ToyContinuousBlackBox(function_name=FUNCTION_NAME, n_dimensions=N_DIMS)

    return torch.from_numpy(f(x.numpy(force=True))).to(torch.get_default_dtype())


def compute_objective_function_optima() -> torch.Tensor:
    f = ToyContinuousBlackBox(function_name=FUNCTION_NAME, n_dimensions=N_DIMS)
    return torch.from_numpy(f.function.optima).to(torch.get_default_dtype())
