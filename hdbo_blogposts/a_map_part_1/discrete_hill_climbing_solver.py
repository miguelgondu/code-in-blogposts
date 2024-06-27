import numpy as np
from poli_baselines.core.abstract_solver import AbstractSolver, AbstractBlackBox

from utils import load_alphabet, run_experiment


class DiscreteHillClimbing(AbstractSolver):
    def __init__(
        self,
        black_box: AbstractBlackBox,
        alphabet: list[str],
        max_sequence_length: int,
        x0: np.ndarray | None = None,
        y0: np.ndarray | None = None,
    ):
        super().__init__(black_box, x0, y0)
        self.alphabet = alphabet
        self.max_sequence_length = max_sequence_length

        self.history = {
            "x": [x_i.reshape(1, -1) for x_i in x0] if x0 is not None else [],
            "y": [y_i.reshape(1, -1) for y_i in y0] if y0 is not None else [],
            "best_y": (
                [np.max(y0[: i + 1]).reshape(1, -1) for i in range(len(y0))]
                if y0 is not None
                else []
            ),
        }

    def solve(
        self,
        max_iter: int = 100,
        n_initial_points: int = 0,
        seed: int | None = None,
    ) -> None:
        if seed is not None:
            np.random.seed(seed)

        for _ in range(max_iter):
            best_idx_so_far = np.argmax(self.history["y"])
            best_sequence_so_far = self.history["x"][best_idx_so_far].reshape(1, -1)

            # Randomly select a position to mutate
            position_to_mutate = np.random.randint(self.max_sequence_length)

            # Randomly select a new token
            new_token = np.random.choice(self.alphabet)

            # Mutate the best sequence so far
            new_sequence = best_sequence_so_far.copy()
            new_sequence[0, position_to_mutate] = new_token

            # Evaluate the new sequence
            val = self.black_box(new_sequence)

            # Update the history
            self.history["x"].append(new_sequence)
            self.history["y"].append(val)
            self.history["best_y"].append(max(val, self.history["best_y"][-1]))

            print(f"Sequence: {new_sequence}, Value: {val}")


if __name__ == "__main__":
    from poli.repository import AlbuterolSimilarityBlackBox

    black_box = AlbuterolSimilarityBlackBox(string_representation="SELFIES")

    alphabet = load_alphabet()
    max_sequence_length = 70
    seed = 42

    np.random.seed(seed)

    x0_ = []
    for _ in range(10):
        x0_.append(np.random.choice(alphabet, max_sequence_length))

    x0 = np.array(x0_)
    y0 = black_box(x0)

    solver = DiscreteHillClimbing(
        black_box, alphabet, max_sequence_length, x0=x0, y0=y0
    )

    run_experiment(
        solver,
        max_iter=500,
        seed=seed,
        experiment_name="discrete_hill_climbing",
    )
