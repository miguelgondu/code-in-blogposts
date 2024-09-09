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

    print(f"Number of sequences in the seed data: {len(x_in_seed_filtered)}")
    print(f"Number of sequences in the RASP seed data: {len(x_rasp)}")
    print(f"Number of labels in the RASP seed data: {len(all_y_rasp)}")
    assert (x_in_seed_filtered == x_rasp).all()

    all_y_foldx = df["stability"].values

    with open("closest_pdbs_clean.json") as f:
        closest_pdbs = json.load(f)
        df_closest_pdbs = pd.DataFrame(closest_pdbs)

    closest_pdb_to_sequence = [
        df_closest_pdbs[df_closest_pdbs["sequence"] == seq]["closest_pdb"].values[0]
        for seq in x_in_seed_filtered
    ]

    for closest_pdb_ in df_closest_pdbs["closest_pdb"].unique():
        print(closest_pdb_)
        sequences = df_closest_pdbs[df_closest_pdbs["closest_pdb"] == closest_pdb_][
            "sequence"
        ].values

        y_foldx = np.array(
            [
                df[df["foldx_seq"] == seq]["stability"].values[0]
                for seq in sequences
                if len(seq) != 227
            ]
        )
        y_rasp = np.array(
            [
                rasp_seed_data["y0"][np.where(x_rasp == seq)[0][0]]
                for seq in sequences
                if len(seq) != 227
            ]
        ).flatten()
        if len(y_foldx) == 0:
            continue
        corr = np.corrcoef(y_foldx, y_rasp)[0, 1]

        slope, intercept, r_value, p_value, std_err = linregress(
            y_foldx, y_rasp, alternative="greater"
        )
        domain = np.linspace(min(y_foldx), max(y_foldx), 100)
        plt.plot(domain, slope * domain + intercept, color="red")

        print(f"Slope ({closest_pdb_}): {slope}")
        print(f"Intercept ({closest_pdb_}): {intercept}")
        print(f"R-squared ({closest_pdb_}): {r_value ** 2}")
        print(f"P-value ({closest_pdb_}): {p_value}")
        sns.scatterplot(
            x=y_foldx,
            y=y_rasp,
            label=f"{closest_pdb_} (corr: {corr:.2f}, slope: {slope:.2f})",
        )

    corr = np.corrcoef(all_y_foldx, all_y_rasp)[0, 1]
    plt.xlabel("FoldX stability")
    plt.ylabel("Additive RASP stability")
    plt.title(f"Correlation: {corr:.2f}")
    plt.legend()

    slope, intercept, r_value, p_value, std_err = linregress(
        all_y_foldx, all_y_rasp, alternative="greater"
    )
    domain = np.linspace(min(all_y_foldx), max(all_y_foldx), 100)
    plt.plot(domain, slope * domain + intercept, color="red")

    print(f"Slope: {slope}")
    print(f"Intercept: {intercept}")
    print(f"R-squared: {r_value ** 2}")
    print(f"P-value: {p_value}")

    plt.show()
