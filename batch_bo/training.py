from pathlib import Path

import torch
import gpytorch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll

ROOT_DIR = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


def train_model_using_gradient_descent(
    model: gpytorch.models.ExactGP,
    likelihood: gpytorch.likelihoods.GaussianLikelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    n_points: int,
    n_dimensions: int,
    patience: int = 50,
    model_name: str = "model",
    max_nr_iterations: int = 1000,
) -> tuple[gpytorch.models.ExactGP, gpytorch.likelihoods.GaussianLikelihood, int, bool]:
    """Train the model."""
    # Set the model and likelihood to training mode
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Training loop
    # This training loop uses early stopping.
    training_had_nans = False
    best_loss = float("inf")
    patience_counter = 0
    for i in range(max_nr_iterations):
        # Zero gradients from previous iteration
        optimizer.zero_grad()

        # Output from model
        output = model(train_x)

        # Calculate loss and backpropagate gradients
        loss = -mll(output, train_y)
        loss.backward()

        # Computing the test loss
        with torch.no_grad():
            model.eval()
            likelihood.eval()
            mll.eval()

            test_loss = float("inf")
            try:
                test_output = model(test_x)
                test_loss = -mll(test_output, test_y)
            except Exception as e:
                print(f"Exception {e} occurred at test iteration {i + 1}.")
                pass

            model.train()
            likelihood.train()
            mll.train()

        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0

            # Save the model
            best_model_state_dict = model.state_dict()
            best_likelihood_state_dict = likelihood.state_dict()

            torch.save(
                best_model_state_dict,
                MODELS_DIR
                / f"model_{model_name}_npoints_{n_points}_ndim_{n_dimensions}.pt",
            )

        else:
            patience_counter += 1

        if patience_counter > patience:
            print(f"Patience reached at iteration {i + 1}. Stopping training early.")
            break

        if torch.isnan(loss):
            print(f"Loss is NaN at iteration {i + 1}.")
            training_had_nans = True
            break

        optimizer.step()

        print(
            f"Nr. points: {n_points} - Dim: {n_dimensions} - Iteration {i + 1}/{max_nr_iterations} - Train loss: {loss.item()} - Test loss: {test_loss}"
        )

    # Load the best model
    model.load_state_dict(best_model_state_dict)
    likelihood.load_state_dict(best_likelihood_state_dict)

    # Set the model and likelihood to evaluation mode
    model.eval()
    likelihood.eval()

    return model, likelihood, i, training_had_nans


def train_model_using_botorch_utils(model: SingleTaskGP) -> SingleTaskGP:
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()

    return model
