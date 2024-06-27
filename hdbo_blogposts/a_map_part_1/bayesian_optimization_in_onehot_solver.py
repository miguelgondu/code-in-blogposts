import json
from typing import Callable

import torch
import numpy as np

from poli_baselines.solvers.bayesian_optimization.vanilla_bo_hvarfner import (
    VanillaBOHvarfner,
)
from poli.core.abstract_black_box import AbstractBlackBox

from utils import load_alphabet, run_experiment


def from_string_to_one_hot(x: np.ndarray, alphabet, sequence_length):
    x_int = np.array([[alphabet.index(c) for c in s] for s in x])
    x_onehot = np.zeros((x_int.shape[0], sequence_length, len(alphabet)))
    for i, x_i in enumerate(x_int):
        x_onehot[i, np.arange(len(x_i)), x_i] = 1

    return x_onehot


def from_one_hot_to_string(x: np.ndarray, alphabet):
    x_int = np.argmax(x, axis=-1)
    x_str = ["".join([alphabet[i] for i in x_i]) for x_i in x_int]
    return np.array(x_str)


def from_discrete_black_box_to_one_hot(
    black_box: AbstractBlackBox,
    alphabet: list[str],
    sequence_length: int,
) -> Callable[[np.ndarray], np.ndarray]:
    def continuous_black_box(x_onehot_flattened: np.ndarray) -> np.ndarray:
        x_onehot = x_onehot_flattened.reshape(-1, sequence_length, len(alphabet))

        # Transform x from one-hot to string
        # 1st, we need to transform x from one-hot to integers
        x_int = np.argmax(x_onehot, axis=-1)

        # Then, we transform the integers to strings
        x_str = ["".join([alphabet[i] for i in x_i]) for x_i in x_int]
        x_str = np.array(x_str)
        return black_box(x_str)

    return continuous_black_box


if __name__ == "__main__":
    from poli.repository import AlbuterolSimilarityBlackBox

    black_box = AlbuterolSimilarityBlackBox(string_representation="SELFIES")

    alphabet = load_alphabet()
    max_sequence_length = 70
    seed = 42

    np.random.seed(seed)
    torch.manual_seed(seed)

    x0_ = []
    for _ in range(10):
        x0_.append(np.random.choice(alphabet, max_sequence_length))

    x0 = np.array(x0_)
    x0_onehot = from_string_to_one_hot(x0, alphabet, max_sequence_length)
    y0 = black_box(x0)

    continuous_function = from_discrete_black_box_to_one_hot(
        black_box, alphabet, max_sequence_length
    )

    continuous_function.info = black_box.info
    continuous_function.num_workers = 6

    solver = VanillaBOHvarfner(
        continuous_function,
        x0=x0_onehot.reshape(-1, max_sequence_length * len(alphabet)),
        y0=y0,
        bounds=[(0, 1)] * (max_sequence_length * len(alphabet)),
    )

    solver.solve(max_iter=500)

    df = solver.ax_client.get_trials_data_frame()

    xs = [df[f"x{i}"].values for i in range(max_sequence_length * len(alphabet))]
    xs = np.array(xs).T
    xs = xs.reshape(-1, max_sequence_length, len(alphabet))
    xs_str = from_one_hot_to_string(xs, alphabet)
    ys = df["albuterol_similarity"].values
    best_ys = np.maximum.accumulate(ys)

    history = {
        "x": [[x_i] for x_i in xs_str.tolist()],
        "y": [[y_i] for y_i in ys.tolist()],
        "best_y": [[y_i] for y_i in best_ys.tolist()],
    }

    with open("one_hot_bo_history.json", "w") as f:
        json.dump(history, f)
