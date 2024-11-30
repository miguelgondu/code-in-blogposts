from pathlib import Path

import torch
import gpytorch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll

ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def train_model_using_botorch_utils(model: SingleTaskGP) -> SingleTaskGP:
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    mll = fit_gpytorch_mll(mll)
    assert not mll.training

    model.eval()
    return model
