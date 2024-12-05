from time import time
from pathlib import Path

import pandas as pd
import numpy as np

from poli.repository import RaspProblemFactory

THIS_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    # Loading up the seed data:
    df = pd.read_csv(THIS_DIR / "proxy_rfp_seed_data.csv")
    x_in_seed = df["foldx_seq"].values
    x_in_seed_filtered = [x for x in x_in_seed if len(x) != 227]

    RFP_PDBS_DIR = THIS_DIR / "rfp_pdbs"
    ALL_PDBS = list(RFP_PDBS_DIR.rglob("**/*.pdb"))
    problem = RaspProblemFactory().create(
        wildtype_pdb_path=ALL_PDBS,
        additive=True,
        chains_to_keep=[p.parent.name.split("_")[1] for p in ALL_PDBS],
    )

    print(problem)
    f, x0 = problem.black_box, problem.x0

    start_time = time()
    y0_for_filtered_seed_data = f(np.array(x_in_seed_filtered))
    print(f"Time taken: {time() - start_time:.2f}s")

    # Save the seed data
    np.savez(
        THIS_DIR / "rasp_seed_data.npz",
        x0=x_in_seed_filtered,
        y0=y0_for_filtered_seed_data,
    )
