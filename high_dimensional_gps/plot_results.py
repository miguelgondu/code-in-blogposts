from pathlib import Path
import json
from typing import Literal

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import LinearSegmentedColormap

myColors = (
    (198 / 255, 210 / 255, 210 / 255, 1.0),
    (69 / 255, 71 / 255, 74 / 255, 1.0),
)
cmap = LinearSegmentedColormap.from_list("Custom", myColors, len(myColors))

# sns.set_style("darkgrid")
sns.set_theme(font_scale=1.3)


# Building the dataset
def build_dataset(pattern: str) -> pd.DataFrame:
    df_rows = []
    for n_points in [50, 100, 500, 1000, 1500, 2000, 2500, 5000, 7500, 10000]:
        for n_dimensions_exponent in range(11):
            n_dimensions = 2**n_dimensions_exponent
            # load the data
            filename = f"./results/{pattern}_n_dimensions_{n_dimensions:05d}_n_points_{n_points:05d}.json"
            if not Path(filename).exists():
                continue

            with open(filename, "r") as fp:
                data = json.load(fp)

            predictions = data["predictions"]

            defaulted_to_mean = max(np.abs(np.mean(predictions) - predictions)) < 1e-1
            print(
                f"n_dimensions: {n_dimensions}, n_points: {n_points} - defaulted_to_mean: {defaulted_to_mean} - {max(np.mean(predictions) - predictions)}"
            )

            df_rows.append(
                {
                    "n_dimensions": n_dimensions,
                    "n_points": n_points,
                    # "actual_values": data["actual_values"],
                    # "predictions": data["predictions"],
                    "training_had_nans": data["training_had_nans"],
                    "defaulted_to_mean": defaulted_to_mean,
                    "correlation": (
                        data["correlation"] if data["correlation"] != "NaN" else np.NaN
                    ),
                    "rsme": np.sqrt(
                        np.mean(
                            np.square(
                                np.array(data["actual_values"])
                                - np.array(data["predictions"])
                            )
                        )
                    ),
                    "mae": np.mean(
                        np.abs(
                            np.array(data["actual_values"])
                            - np.array(data["predictions"])
                        )
                    ),
                    "mape": np.mean(
                        np.abs(
                            np.array(data["actual_values"])
                            - np.array(data["predictions"])
                        )
                        / np.array(data["actual_values"])
                    ),
                }
            )

    df = pd.DataFrame(df_rows)

    return df


df_w_ard = build_dataset("values_and_predictions_w_ard_True")
df_wo_ard = build_dataset("values_and_predictions_w_ard_False")
df_logprior = build_dataset("values_and_predictions_log_normal")
df_logprior.dropna(axis=0, inplace=True)

# Plotting the results
figure_path = Path("figures")
figure_path.mkdir(parents=True, exist_ok=True)


# Plotting the correlation
# sns.lineplot(
#     data=df_w_ard,
#     x="n_points",
#     y="correlation",
#     hue="n_dimensions",
#     style="training_had_nans",
#     markers=True,
#     dashes=False,
#     markersize=10,
#     alpha=0.85,
# )
# cmap = sns.color_palette("rocket", 2)
def plot_predictions_defaulted_to_mean(
    df: pd.DataFrame, comment: str = ""
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    sns.heatmap(
        1.0
        - df.pivot(
            index="n_points", columns="n_dimensions", values="defaulted_to_mean"
        ).astype(float),
        # annot=True,
        # fmt=".4f",
        cmap=cmap,
        linewidths=0.5,
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_title(f"{comment}Did the model learn anything?")
    ax.set_xlabel("nr. dimensions")
    ax.set_ylabel("nr. points")
    # cmap = sns.color_palette("rocket", 2)
    colorbar = ax.collections[0].colorbar
    # r = colorbar.vmax - colorbar.vmin
    r = 1  # 1 - 0, we always know what the values are
    colorbar.set_ticks([0 + r / 2 * (0.5 + i) for i in range(2)])
    colorbar.set_ticklabels(["no :(", "yes :)"])
    fig.tight_layout()

    return fig, ax


def plot_training_had_nans(df: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    sns.heatmap(
        df_w_ard.pivot(
            index="n_points", columns="n_dimensions", values="training_had_nans"
        ),
        # annot=True,
        # fmt="1d",
        cmap=cmap,
        linewidths=0.5,
        # vmin=0,
        # vmax=1,
        ax=ax,
        # cbar=False,
    )
    colorbar = ax.collections[0].colorbar
    # r = colorbar.vmax - colorbar.vmin
    r = 1
    colorbar.set_ticks([colorbar.vmin + r / 2 * (0.5 + i) for i in range(2)])
    colorbar.set_ticklabels(["no :)", "yes :("])
    ax.set_title("Did we get NaNs during training?")
    ax.set_xlabel("nr. dimensions")
    ax.set_ylabel("nr. points")
    fig.tight_layout()

    return fig, ax


def plot_continuous_value(
    df: pd.DataFrame, continuous_value: Literal["mae", "mape"]
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    sns.heatmap(
        df.pivot(index="n_points", columns="n_dimensions", values=continuous_value),
        annot=True,
        fmt=".2f",
        ax=ax,
        cbar=False,
        linewidths=0.5,
        # cmap="viridis",
        # vmin=0,
        # vmax=1,
    )
    ax.set_title(f"{continuous_value}")
    ax.set_xlabel("nr. dimensions")
    ax.set_ylabel("nr. points")
    fig.tight_layout()

    return fig, ax


plot_predictions_defaulted_to_mean(df_logprior, comment="Log prior: ")
plot_predictions_defaulted_to_mean(df_w_ard, comment="Vanilla: ")
# plot_training_had_nans(df_logprior)
plot_continuous_value(df_logprior, "mae")
plot_continuous_value(df_logprior, "mape")

plt.show()
