from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF

from scipy.stats import multivariate_normal

from plot_utilities import plot_comparison_between_actual_and_predicted_values


def shifted_sphere(x, offset):
    """A shifted sphere function."""
    return np.sum((x - offset) ** 2, axis=1)


def save_comparison(n_points, n_dimensions, seed: int = 123):
    np.random.seed(seed)

    # Set up training data
    training_dist = multivariate_normal(
        mean=np.zeros(n_dimensions), cov=1.0 * np.eye(n_dimensions)
    )
    train_x = training_dist.rvs(n_points).reshape(n_points, n_dimensions)

    random_offset = np.random.normal(size=(1, n_dimensions))

    train_y = (shifted_sphere(train_x, offset=random_offset)).flatten()
    random_noise = np.random.normal(size=train_y.shape)
    train_y += 0.25 * random_noise

    # Normalize the data
    # normalizer = Normalizer()
    # train_x = normalizer.fit_transform(train_x)

    scaler_y = MinMaxScaler()
    train_y = (scaler_y.fit_transform(train_y.reshape(-1, 1))).flatten()

    # Train the model
    kernel = 1.0 * (RBF(length_scale=np.ones(n_dimensions)) + WhiteKernel())
    gpr = GaussianProcessRegressor(kernel=kernel).fit(train_x, train_y)

    # Initialize the model and likelihood

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
    another_sample_from_training_dist = training_dist.rvs(n_points).reshape(
        n_points, n_dimensions
    )
    test_prediction_mean, std_dev = gpr.predict(
        another_sample_from_training_dist, return_std=True
    )
    test_predictions_95 = 1.96 * std_dev

    actual_values = shifted_sphere(
        another_sample_from_training_dist, offset=random_offset
    )
    actual_values = scaler_y.transform(actual_values.reshape(-1, 1)).flatten()

    # make sure the folder where the figure is to be saved exists
    figure_path = Path("figures") / "sklearn"
    figure_path.mkdir(parents=True, exist_ok=True)

    plot_comparison_between_actual_and_predicted_values(
        actual_values=actual_values,
        predicted_values=test_prediction_mean,
        error_bars=test_predictions_95,
        figure_path=figure_path
        / f"comparison_shifted_sphere_w_ard_n_dimensions_{n_dimensions:05d}_n_points_{n_points:05d}.png",
        n_dimensions=n_dimensions,
        n_points=n_points,
    )
    # plt.tight_layout()

    # plt.show()
    plt.close("all")


if __name__ == "__main__":
    for n_points in [50, 100, 200, 500]:
        for n_dimensions in list(range(1, 32)) + [32, 64, 128, 256, 512, 1024]:
            save_comparison(n_points=n_points, n_dimensions=n_dimensions)
