from pathlib import Path

import gpytorch  # noqa
import numpy as np  # noqa

PROJECT_DIR = Path(__file__).parent.resolve()

FIGURES_DIR = PROJECT_DIR / "figures"

FUNCTION_NAME = "cross_in_tray"
N_DIMS = 2
TOTAL_BUDGET = 50

DEFAULT_KERNEL = gpytorch.kernels.ScaleKernel(
    gpytorch.kernels.MaternKernel(
        ard_num_dims=N_DIMS,
        lengthscale_prior=gpytorch.priors.LogNormalPrior(
            np.sqrt(2.0) + np.log(N_DIMS) * 0.5, np.sqrt(3.0)
        ),
    )
)
# DEFAULT_KERNEL = gpytorch.kernels.ScaleKernel(
#     gpytorch.kernels.RBFKernel(ard_num_dims=N_DIMS)
# )
SEED = 0
