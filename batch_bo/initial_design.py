import torch
import torch.quasirandom

from objective_function import objective_function

from dataset import Dataset


def compute_initial_design_using_sobol(
    n_points: int,
    n_dimension: int = 2,
    seed: int = None,
    bounds: tuple[float, float] = (-10.0, 10.0),
) -> Dataset:

    sobol_engine = torch.quasirandom.SobolEngine(
        dimension=n_dimension,
        seed=seed,
    )
    lb, ub = bounds
    initial_design = sobol_engine.draw(n_points) * (ub - lb) + lb

    y_values = objective_function(initial_design)

    return Dataset(X=initial_design, y=y_values)
