from pathlib import Path
from typing import Union

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


def plot_comparison_between_actual_and_predicted_values(
    actual_values: Union[np.ndarray, torch.Tensor],
    predicted_values: Union[np.ndarray, torch.Tensor],
    error_bars: Union[np.ndarray, torch.Tensor],
    figure_path: Path,
    n_dimensions: int,
    n_points: int,
):
    if isinstance(actual_values, torch.Tensor):
        actual_values = actual_values.numpy(force=True)

    if isinstance(predicted_values, torch.Tensor):
        predicted_values = predicted_values.numpy(force=True)

    if isinstance(error_bars, torch.Tensor):
        error_bars = error_bars.numpy(force=True)

    min_ = min(np.concatenate((predicted_values, actual_values)))
    max_ = max(np.concatenate((predicted_values, actual_values)))

    fig = plt.figure(figsize=(8, 8))
    plt.errorbar(
        x=actual_values,
        y=predicted_values,
        yerr=error_bars,
        fmt=".k",
        label="mean predictions vs. actual values",
        markersize=10,
        markerfacecolor="white",
        markeredgecolor="gray",
        alpha=0.85,
        # errorbar=test_predictions_95.numpy(force=True),
    )

    padding = 0.05 * (max_ - min_)
    plt.plot(
        np.linspace(min_ - padding, max_ + padding, 100),
        np.linspace(min_ - padding, max_ + padding, 100),
        color="green",
        label="y=x",
        linestyle="--",
        alpha=0.5,
    )
    plt.xlabel("actual values")
    plt.ylabel("mean predictions")
    plt.title(f"nr. dimensions: {n_dimensions}, nr. points: {n_points}")
    plt.legend()

    plt.xlim(min_ - padding, max_ + padding)
    plt.ylim(min_ - padding, max_ + padding)
    plt.tight_layout()

    fig.savefig(figure_path, dpi=300)
