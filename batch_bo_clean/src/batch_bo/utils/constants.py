from pathlib import Path

import gpytorch
import numpy as np

ROOT_DIR = Path(__file__).parent.parent.parent.parent.resolve()
FIGURES_DIR = ROOT_DIR / "figures"

FUNCTION_NAME = "himmelblau"
N_DIMS = 2
TOTAL_BUDGET = 50
LIMITS = (-5.0, 5.0)
RESOLUTION = 100
INITIAL_DESIGN_SIZE = 10

# DEFAULT_KERNEL_GPYTORCH = gpytorch.kernels.ScaleKernel(
#     gpytorch.kernels.MaternKernel(
#         ard_num_dims=N_DIMS,
#         lengthscale_prior=gpytorch.priors.LogNormalPrior(
#             np.sqrt(2.0) + np.log(1.0) * 0.5 + 0.5, np.sqrt(3.0)
#         ),
#     )
# )
DEFAULT_KERNEL_GPYTORCH = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
SEED = None
