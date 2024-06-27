"""Implements the log-normal prior proposed by Hvarfner et al in https://arxiv.org/abs/2402.02229."""

from pathlib import Path
import json

import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from plot_utilities import plot_comparison_between_actual_and_predicted_values

from torch.distributions import MultivariateNormal

from gpytorch.priors import LogNormalPrior

sns.set_style("darkgrid")

from torch.distributions import constraints
from torch.distributions.normal import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import ExpTransform

__all__ = ["LogNormal"]


# class LogNormal(TransformedDistribution):
#     r"""
#     Creates a log-normal distribution parameterized by
#     :attr:`loc` and :attr:`scale` where::

#         X ~ Normal(loc, scale)
#         Y = exp(X) ~ LogNormal(loc, scale)

#     Example::

#         >>> # xdoctest: +IGNORE_WANT("non-deterinistic")
#         >>> m = LogNormal(torch.tensor([0.0]), torch.tensor([1.0]))
#         >>> m.sample()  # log-normal distributed with mean=0 and stddev=1
#         tensor([ 0.1046])

#     Args:
#         loc (float or Tensor): mean of log of distribution
#         scale (float or Tensor): standard deviation of log of the distribution
#     """

#     arg_constraints = {"loc": constraints.real, "scale": constraints.positive}
#     support = constraints.positive
#     has_rsample = True

#     def __init__(self, loc, scale, validate_args=None):
#         base_dist = Normal(loc, scale, validate_args=validate_args)
#         super().__init__(base_dist, ExpTransform(), validate_args=validate_args)

#     def expand(self, batch_shape, _instance=None):
#         new = self._get_checked_instance(LogNormal, _instance)
#         return super().expand(batch_shape, _instance=new)

#     @property
#     def loc(self):
#         return self.base_dist.loc

#     @property
#     def scale(self):
#         return self.base_dist.scale

#     @property
#     def mean(self):
#         return (self.loc + self.scale.pow(2) / 2).exp()

#     @property
#     def mode(self):
#         return (self.loc - self.scale.square()).exp()

#     @property
#     def variance(self):
#         scale_sq = self.scale.pow(2)
#         return scale_sq.expm1() * (2 * self.loc + scale_sq).exp()

#     def entropy(self):
#         return self.base_dist.entropy() + self.loc


def shifted_sphere(x, offset):
    """A shifted sphere function."""
    return torch.sum((x - offset) ** 2, dim=1)


# Define the Gaussian Process model
class ExactGPModelWithLogNormalPrior(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModelWithLogNormalPrior, self).__init__(
            train_x, train_y, likelihood
        )
        _, n_dimensions = train_x.shape

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=n_dimensions,
                lengthscale_prior=gpytorch.priors.LogNormalPrior(
                    np.log(n_dimensions) / 2, 1.0
                ),
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def save_comparison(n_points: int, n_dimensions: int, seed: int = 123):
    torch.manual_seed(seed)

    # Set up training data
    training_dist = MultivariateNormal(
        torch.zeros(n_dimensions), 1.0 * torch.eye(n_dimensions)
    )
    train_x = training_dist.sample((n_points,))

    random_offset = torch.rand(1, n_dimensions)

    train_y = (shifted_sphere(train_x, offset=random_offset)).flatten()
    random_noise = torch.randn_like(train_y)
    train_y += 0.25 * random_noise

    # Normalize the data
    # normalizer = Normalizer()
    # train_x = normalizer.fit_transform(train_x)

    scaler_y = MinMaxScaler()
    train_y = (
        torch.from_numpy(scaler_y.fit_transform(train_y.reshape(-1, 1)))
        .flatten()
        .to(torch.float32)
    )

    # Initialize the model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModelWithLogNormalPrior(train_x, train_y, likelihood)

    # Set the model and likelihood to training mode
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Training loop
    training_iterations = 1000
    training_had_nans = False
    for i in range(training_iterations):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calculate loss and backpropagate gradients
        loss = -mll(output, train_y)
        loss.backward()

        if torch.isnan(loss):
            print(f"Loss is NaN at iteration {i + 1}.")
            training_had_nans = True
            break

        optimizer.step()

        print(
            f"Nr. points: {n_points} - Dim: {n_dimensions} - Iteration {i + 1}/{training_iterations} - Loss: {loss.item()}"
        )

    # Set the model and likelihood to evaluation mode
    model.eval()
    likelihood.eval()

    # Make predictions with the trained model
    # test_x = torch.linspace(train_x.min().item(), train_x.max().item(), 51).reshape(
    #     -1, n_dimensions
    # )
    # with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #     predictions = likelihood(model(test_x))

    # # Plot the results
    # plt.figure(figsize=(8, 6))
    # plt.scatter(
    #     train_x.numpy(),
    #     train_y.numpy(),
    #     color="black",
    #     marker="o",
    #     label="Training data",
    # )
    # plt.plot(
    #     test_x.numpy(), predictions.mean.numpy(), color="blue", label="Mean Prediction"
    # )
    # plt.fill_between(
    #     test_x.flatten().numpy(),
    #     predictions.mean.numpy() - 1.96 * predictions.stddev.numpy(),
    #     predictions.mean.numpy() + 1.96 * predictions.stddev.numpy(),
    #     color="lightblue",
    #     alpha=0.5,
    #     label="95% Confidence Interval",
    # )
    # plt.title("Gaussian Process Regression with GPyTorch")
    # plt.xlabel("Input")
    # plt.ylabel("Output")
    # plt.legend()

    # Comparing predictions w. actual values in a random sample
    n_test_points = 50
    another_sample_from_training_dist = training_dist.sample((n_test_points,))

    test_predictions = likelihood(model(another_sample_from_training_dist))
    test_predictions_mean = test_predictions.mean
    test_predictions_95 = 1.96 * test_predictions.stddev

    actual_values = shifted_sphere(
        another_sample_from_training_dist, offset=random_offset
    )
    actual_values = torch.from_numpy(
        scaler_y.transform(actual_values.reshape(-1, 1))
    ).flatten()

    # make sure the folder where the figure is to be saved exists
    figure_path = Path("figures") / "gpytorch"
    figure_path.mkdir(parents=True, exist_ok=True)

    plot_comparison_between_actual_and_predicted_values(
        actual_values=actual_values,
        predicted_values=test_predictions_mean,
        error_bars=test_predictions_95,
        figure_path=figure_path
        / f"comparison_shifted_sphere_log_normal_n_dimensions_{n_dimensions:05d}_n_points_{n_points:05d}.png",
        n_dimensions=n_dimensions,
        n_points=n_points,
    )
    # plt.tight_layout()

    # Saving the actual values and the predictions.
    correlation = np.corrcoef(
        actual_values.numpy(force=True),
        test_predictions_mean.numpy(force=True),
    )[0, 1]

    if np.isnan(correlation):
        correlation = "NaN"

    with open(
        f"./results/values_and_predictions_log_normal_n_dimensions_{n_dimensions:05d}_n_points_{n_points:05d}.json",
        "w",
    ) as fp:
        json.dump(
            {
                "actual_values": actual_values.numpy(force=True).tolist(),
                "predictions": test_predictions_mean.numpy(force=True).tolist(),
                "error_bars": test_predictions_95.numpy(force=True).tolist(),
                "training_had_nans": training_had_nans,
                "defaulted_to_mean": len(torch.unique(test_predictions_mean)) == 1,
                "correlation": correlation,
                "seed": seed,
            },
            fp,
        )

    # plt.show()
    plt.close("all")


if __name__ == "__main__":
    seed = np.random.randint(0, 100000)
    # for n_points in [2500, 5000, 7500, 10000]:
    for n_points in [
        # 50,
        # 100,
        # 500,
        # 1000,
        # 1500,
        # 2000,
        # 2500,
        # 5000,
        7500,
        10000,
    ]:
        for n_dimensions_exponent in range(11):
            save_comparison(
                n_points=n_points,
                n_dimensions=2**n_dimensions_exponent,
                seed=seed,
            )
