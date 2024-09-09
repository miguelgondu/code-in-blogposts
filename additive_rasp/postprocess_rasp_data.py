import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress

if __name__ == "__main__":
    df = pd.read_csv("proxy_rfp_seed_data.csv")

    df = df[df["foldx_seq"].apply(len) != 227]
    x_in_seed_filtered = df["foldx_seq"].values

    rasp_seed_data = np.load("rasp_seed_data.npz")
    x_rasp = rasp_seed_data["x0"]
    all_y_rasp = rasp_seed_data["y0"].flatten()

    with open("closest_pdbs_clean.json") as f:
        closest_pdbs = json.load(f)
        df_closest_pdbs = pd.DataFrame(closest_pdbs)

    closest_pdb_to_sequence_foldx = np.array(
        [
            df_closest_pdbs[df_closest_pdbs["sequence"] == seq]["closest_pdb"].values[0]
            for seq in x_in_seed_filtered
        ]
    )
    closest_pdb_to_sequence_rasp = np.array(
        [
            df_closest_pdbs[df_closest_pdbs["sequence"] == seq]["closest_pdb"].values[0]
            for seq in x_rasp
        ]
    )

    assert (closest_pdb_to_sequence_foldx == closest_pdb_to_sequence_rasp).all()

    final_df = pd.DataFrame(
        {
            "sequence": x_in_seed_filtered,
            "closest_pdb": closest_pdb_to_sequence_foldx,
            "foldx_score": df["stability"].values,
            "rasp_score": all_y_rasp,
        }
    )

    final_df.to_csv("comparison_foldx_rasp.csv", index=False)
