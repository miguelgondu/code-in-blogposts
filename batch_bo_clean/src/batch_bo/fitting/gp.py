from pathlib import Path

import torch
import gpytorch

import jax.random as jr
import gpjax as gpx
import optax as ox

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll

from gpytorch.models import ExactGP

from batch_bo.models.gp import ExactGPModelJax, ExactGPScikitLearn

ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def train_model_in_scikit_learn(model: ExactGPScikitLearn) -> ExactGPScikitLearn:
    return model


def train_model_using_botorch_utils(model: SingleTaskGP) -> SingleTaskGP:
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    mll = fit_gpytorch_mll(mll)
    assert not mll.training

    model.eval()
    return model


def train_exact_gp_using_gradient_descent(
    model: ExactGP, max_nr_iterations: int = 500
) -> ExactGP:
    # Set the model and likelihood to training mode
    likelihood = model.likelihood

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Training loop
    # This training loop uses early stopping.
    for i in range(max_nr_iterations):
        # Zero gradients from previous iteration
        optimizer.zero_grad()

        # Output from model
        output = model(model.train_x)

        # Calculate loss and backpropagate gradients
        loss = -mll(output, model.train_y)
        loss.backward()

        optimizer.step()

        print(
            f"Iteration {i + 1}/{max_nr_iterations} - "
            f"Train loss: {loss.item():0.3f} - "
            f"Lengthscales: {model.covar_module.base_kernel.lengthscale.numpy(force=True)}"
        )

    # Set the model and likelihood to evaluation mode
    model.eval()
    likelihood.eval()

    return model


def train_exact_gp_jax(
    model: ExactGPModelJax,
    max_nr_iterations: int = 500,
    seed: int = 0,
) -> gpx:
    posterior = model.compute_posterior(model.train_x, model.train_y)
    optimiser = ox.adam(learning_rate=1e-2)

    key = jr.key(seed)

    opt_posterior, history = gpx.fit(
        model=posterior,
        objective=lambda p, d: -1 * gpx.objectives.conjugate_mll(p, d),
        train_data=model.D,
        optim=optimiser,
        num_iters=max_nr_iterations,
        safe=True,
        key=key,
    )

    return opt_posterior


def fit_gp(dataset: gpx.Dataset, key: jr.PRNGKey) -> gpx.gps.ConjugatePosterior:
    # Construct the prior
    meanf = gpx.mean_functions.Zero()
    kernel = gpx.kernels.RBF()
    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

    # Define a likelihood
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=len(dataset.y))

    # Construct the posterior
    posterior = prior * likelihood

    # Define an optimiser
    optimiser = ox.adam(learning_rate=1e-2)

    # Define the marginal log-likelihood
    opt_posterior, _ = gpx.fit(
        model=posterior,
        objective=lambda p, d: -1.0 * gpx.objectives.conjugate_mll(p, d),
        train_data=dataset,
        optim=optimiser,
        num_iters=500,
        safe=True,
        key=key,
    )

    return opt_posterior
