from pathlib import Path

import numpy as np
import pandas as pd


def some_simulation(
    *, protein: str, molecule: str, rng: np.random.Generator
) -> tuple[float, float]:
    mean = sum([ord(char) for char in protein]) / sum([ord(char) for char in molecule])
    protein_activity = rng.normal(mean, 0.01, size=1)
    accuracy = rng.random(size=1)
    return protein_activity[0], accuracy[0]


def run_experiment_for_protein(protein: str, molecule, seed: int) -> pd.DataFrame:
    """
    This function represents the expernsive experiment you'd
    run for one protein, one molecule (or, more generally,
    exp. configuration) and for one seed.
    """
    rng = np.random.default_rng(seed)
    binding_affinity, accuracy = some_simulation(
        protein=protein,
        molecule=molecule,
        rng=rng,
    )
    return pd.DataFrame(
        [
            {
                "binding_affinity": binding_affinity,
                "accuracy": accuracy,
            }
        ],
    ).rename_axis("repetition")


def run_experiment(
    proteins: list[str],
    molecules: list[str],
    repetitions: int,
) -> None:
    """
    This function represents your entire experiment pipeline,
    which you'll of course run in parallel on your HPC instead
    of sequentially like we do here...

    The idea is that you store your experimental results in a
    specific way:

    data/
        protein=one_protein/
            molecule=one_molecule/
                seed=1/
                    results.csv
                ...
                seed=N/
                    results.csv
            ...
            molecule=final_molecule/
                seed=1/
                    results.csv
                ...
                seed=N/
                    results.csv
        ...
        protein=final_protein/
            molecule=one_molecule/
                seed=1/
                    results.csv
                ...
                seed=N/
                    results.csv
            ...
            molecule=final_molecule/
                seed=1/
                    results.csv
                ...
                seed=N/
                    results.csv

    This is called hive partitioning.
    """

    results_dir = Path(__file__).parent / "results"

    for protein in proteins:
        for molecule in molecules:
            for seed in range(repetitions):
                df = run_experiment_for_protein(protein, molecule, seed)

                # Hive partitioning!
                results_dir_for_experiment = (
                    results_dir
                    / f"protein={protein}"
                    / f"molecule={molecule}"
                    / f"seed={seed}"
                )
                results_dir_for_experiment.mkdir(parents=True, exist_ok=True)

                df.to_csv(results_dir_for_experiment / "results.csv")
