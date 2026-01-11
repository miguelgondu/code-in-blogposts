import json
from pathlib import Path
from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import BaseModel


class ExperimentMetadata(BaseModel):
    run_at: datetime
    model_used: Literal["GAABind", "Dockstring", "PBCNet", "AF3"]
    dataset_used: Literal["PDBbind", "CASF", "Binding MOAD", "BindingDB"]


class ExperimentResults(BaseModel):
    binding_affinity: float
    accuracy: float
    metadata: ExperimentMetadata


def some_simulation(
    *, protein: str, molecule: str, rng: np.random.Generator
) -> ExperimentResults:
    """
    This function represents a simulation of binding affinity using a protein and molecule.
    """
    mean = sum([ord(char) for char in protein]) / sum([ord(char) for char in molecule])
    protein_activity = rng.normal(mean, 0.01, size=1)
    accuracy = rng.random(size=1)

    return ExperimentResults(
        binding_affinity=protein_activity[0],
        accuracy=accuracy[0],
        metadata=ExperimentMetadata(
            run_at=datetime.now(),
            model_used="GAABind",
            dataset_used="PDBbind",
        ),
    )


def run_experiment_for_protein(protein: str, molecule, seed: int) -> ExperimentResults:
    """
    This function represents the expernsive experiment you'd
    run for one protein, one molecule (or, more generally,
    exp. configuration) and for one seed.
    """
    rng = np.random.default_rng(seed)
    res = some_simulation(
        protein=protein,
        molecule=molecule,
        rng=rng,
    )
    return res


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
                res = run_experiment_for_protein(protein, molecule, seed)

                # Hive partitioning!
                results_dir_for_experiment = (
                    results_dir
                    / f"protein={protein}"
                    / f"molecule={molecule}"
                    / f"seed={seed}"
                )
                results_dir_for_experiment.mkdir(parents=True, exist_ok=True)

                with open(results_dir_for_experiment / "results.json", "w") as fp:
                    json.dump(res.model_dump(mode="json"), fp)
