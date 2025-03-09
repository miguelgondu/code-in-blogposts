import torch
import numpy as np

from poli.repository import ToyContinuousBlackBox

from batch_bo.utils.constants import FUNCTION_NAME, N_DIMS, LIMITS, RESOLUTION


def compute_domain() -> torch.Tensor:
    x = np.linspace(*LIMITS, RESOLUTION)
    y = np.linspace(*LIMITS, RESOLUTION)
    xy = np.array(np.meshgrid(x, y)).T.reshape(-1, N_DIMS)
    return torch.from_numpy(xy).to(torch.get_default_dtype())


def objective_function(x: torch.Tensor) -> torch.Tensor:
    f = ToyContinuousBlackBox(function_name=FUNCTION_NAME, n_dimensions=N_DIMS)

    return torch.from_numpy(f(x.numpy(force=True))).to(torch.get_default_dtype())


def compute_objective_function_optima() -> torch.Tensor:
    f = ToyContinuousBlackBox(function_name=FUNCTION_NAME, n_dimensions=N_DIMS)
    return torch.from_numpy(f.function.optima).to(torch.get_default_dtype())
