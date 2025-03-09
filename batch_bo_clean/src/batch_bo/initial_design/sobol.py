import torch
import torch.quasirandom

from batch_bo.functions.objective_function import objective_function

from batch_bo.dataset import Dataset
from batch_bo.utils.constants import LIMITS


def compute_initial_design_using_sobol(
    n_points: int,
    n_dimension: int = 2,
    seed: int = None,
) -> Dataset:

    sobol_engine = torch.quasirandom.SobolEngine(
        dimension=n_dimension,
        seed=seed,
    )
    lb, ub = LIMITS
    initial_design = sobol_engine.draw(n_points) * (ub - lb) + lb

    y_values = objective_function(initial_design)

    return Dataset(X=initial_design, y=y_values)
