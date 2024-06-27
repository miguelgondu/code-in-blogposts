"""Implements a simple example of GP regression on a shifted sphere."""

from pathlib import Path
import json

import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from torch.distributions import MultivariateNormal

from plot_utilities import plot_comparison_between_actual_and_predicted_values
from training import train_model, filter_close_points


sns.set_style("darkgrid")


def shifted_sphere(x, offset):
    """A shifted sphere function."""
    return torch.sum((x - offset) ** 2, dim=1)


# Define the Gaussian Process model
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, w_ard: bool = False):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        if w_ard:
            _, n_dimensions = train_x.shape
        else:
            n_dimensions = None

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=n_dimensions)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def save_comparison(n_points: int, n_dimensions: int, w_ard: bool, seed: int = 123):
    torch.manual_seed(seed)

    # Set up training data
    training_dist = MultivariateNormal(
        torch.zeros(n_dimensions), 1.0 * torch.eye(n_dimensions)
    )
    train_x = training_dist.sample((n_points,))
    test_x = training_dist.sample((n_points // 10,))

    random_offset = torch.rand(1, n_dimensions)

    train_y = (shifted_sphere(train_x, offset=random_offset)).flatten()
    random_noise = torch.randn_like(train_y)
    train_y += 0.25 * random_noise

    test_y = (shifted_sphere(test_x, offset=random_offset)).flatten()
    random_noise = torch.randn_like(test_y)
    test_y += 0.25 * random_noise

    # Normalize the data
    # normalizer = Normalizer()
    # train_x = normalizer.fit_transform(train_x)

    y = torch.cat((train_y, test_y))

    scaler_y = MinMaxScaler().fit(y.reshape(-1, 1))
    train_y = (
        torch.from_numpy(scaler_y.transform(train_y.reshape(-1, 1)))
        .flatten()
        .to(torch.float32)
    )
    test_y = (
        torch.from_numpy(scaler_y.transform(test_y.reshape(-1, 1)))
        .flatten()
        .to(torch.float32)
    )

    # Remove points that are too close to each other
    train_x, train_y = filter_close_points(train_x, train_y)
    test_x, test_y = filter_close_points(test_x, test_y)

    # Initialize the model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood, w_ard=w_ard)

    model, likelihood, _, training_had_nans = train_model(
        model=model,
        likelihood=likelihood,
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        n_points=n_points,
        n_dimensions=n_dimensions,
        max_nr_iterations=int(1e6),
        model_name=f"vanilla_w_ard_{w_ard}",
    )

    # Set the model and likelihood to evaluation mode
    model.eval()
    likelihood.eval()

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
        / f"comparison_shifted_sphere_w_ard_{w_ard}_n_dimensions_{n_dimensions:05d}_n_points_{n_points:05d}.png",
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
        f"./results/values_and_predictions_w_ard_{w_ard}_n_dimensions_{n_dimensions:05d}_n_points_{n_points:05d}.json",
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
                "model_lengthscale": model.covar_module.base_kernel.lengthscale.numpy(
                    force=True
                ).tolist(),
            },
            fp,
        )

    # plt.show()
    plt.close("all")


if __name__ == "__main__":
    from multiprocessing import Pool

    w_ard = True
    seed = np.random.randint(0, 100_000)
    input_args = [
        (n_points, 2**n_dimensions_exponent, w_ard, seed)
        for n_points in [
            50,
            100,
            500,
            1000,
            1500,
            2000,
            2500,
            5000,
            # 7500,
            # 10000,
        ]
        for n_dimensions_exponent in range(11)
    ]
    with Pool(4) as p:
        p.starmap(save_comparison, input_args)
