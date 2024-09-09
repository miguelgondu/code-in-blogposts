import multiprocessing

import pystan
import arviz as az

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

multiprocessing.set_start_method("fork")


def compute_loo(fit, title=""):
    inf_data = az.from_pystan(fit, log_likelihood="log_lik")
    loo = az.loo(inf_data, pointwise=True)

    plt.figure(figsize=(5, 3))
    pareto_k = loo.pareto_k
    plt.plot(pareto_k, "+k")
    plt.axhline(y=1, color="r", linestyle="dotted", label=r"$\hat{k}=1$")
    plt.axhline(y=0.7, color="r", linestyle="--", label=r"$\hat{k}=0.7$")
    plt.axhline(y=0.5, color="r", linestyle="-.", label=r"$\hat{k}=0.5$")
    plt.legend()
    plt.title(r"Pareto $\hat{k}$ " + title)
    plt.xlabel("Observation")
    plt.ylabel(r"$\hat{k}$")
    plt.show()

    return loo


if __name__ == "__main__":
    # Load the data
    df = pd.read_csv("comparison_foldx_rasp.csv")
    unique_pdbs = df["closest_pdb"].unique()

    wildtype_arrays_foldx = [
        df[df["closest_pdb"] == pdb]["foldx_score"].values for pdb in unique_pdbs
    ]
    wildtype_arrays_rasp = [
        df[df["closest_pdb"] == pdb]["rasp_score"].values for pdb in unique_pdbs
    ]
    n_mutations_per_wt = [len(arr) for arr in wildtype_arrays_foldx]
    assert n_mutations_per_wt == [len(arr) for arr in wildtype_arrays_rasp]

    # Fit the model
    model_code = """
    data {
    int<lower=0> n_mutations_per_wt_1;
    int<lower=0> n_mutations_per_wt_2;
    int<lower=0> n_mutations_per_wt_3;
    int<lower=0> n_mutations_per_wt_4;
    int<lower=0> n_mutations_per_wt_5;

    real x_foldx_1[n_mutations_per_wt_1];
    real x_foldx_2[n_mutations_per_wt_2];
    real x_foldx_3[n_mutations_per_wt_3];
    real x_foldx_4[n_mutations_per_wt_4];
    real x_foldx_5[n_mutations_per_wt_5];

    real y_rasp_1[n_mutations_per_wt_1];
    real y_rasp_2[n_mutations_per_wt_2];
    real y_rasp_3[n_mutations_per_wt_3];
    real y_rasp_4[n_mutations_per_wt_4];
    real y_rasp_5[n_mutations_per_wt_5];
    }

    parameters {
    real slope;               // Shared slope
    real offsets[5];     // Intercepts
    real<lower=0> sigma_rasp[5]; // Standard deviations
    }

    model {
    // Priors
    slope ~ normal(0, 10);
    offsets ~ normal(0, 10);

    // Likelihood
    for (n in 1:n_mutations_per_wt_1) {
        y_rasp_1[n] ~ normal(offsets[1] + slope * x_foldx_1[n], sigma_rasp[1]);
    }
    for (n in 1:n_mutations_per_wt_2) {
        y_rasp_2[n] ~ normal(offsets[2] + slope * x_foldx_2[n], sigma_rasp[2]);
    }
    for (n in 1:n_mutations_per_wt_3) {
        y_rasp_3[n] ~ normal(offsets[3] + slope * x_foldx_3[n], sigma_rasp[3]);
    }
    for (n in 1:n_mutations_per_wt_4) {
        y_rasp_4[n] ~ normal(offsets[4] + slope * x_foldx_4[n], sigma_rasp[4]);
    }
    for (n in 1:n_mutations_per_wt_5) {
        y_rasp_5[n] ~ normal(offsets[5] + slope * x_foldx_5[n], sigma_rasp[5]);
    }
    }

    generated quantities {
    vector[n_mutations_per_wt_1 + n_mutations_per_wt_2 + n_mutations_per_wt_3 + n_mutations_per_wt_4 + n_mutations_per_wt_5] log_lik;
    for (n in 1:n_mutations_per_wt_1) {
        log_lik[n] = normal_lpdf(y_rasp_1[n] | offsets[1] + slope * x_foldx_1[n], sigma_rasp[1]);
    }
    for (n in 1:n_mutations_per_wt_2) {
        log_lik[n_mutations_per_wt_1 + n] = normal_lpdf(y_rasp_2[n] | offsets[2] + slope * x_foldx_2[n], sigma_rasp[2]);
    }
    for (n in 1:n_mutations_per_wt_3) {
        log_lik[n_mutations_per_wt_1 + n_mutations_per_wt_2 + n] = normal_lpdf(y_rasp_3[n] | offsets[3] + slope * x_foldx_3[n], sigma_rasp[3]);
    }
    for (n in 1:n_mutations_per_wt_4) {
        log_lik[n_mutations_per_wt_1 + n_mutations_per_wt_2 + n_mutations_per_wt_3 + n] = normal_lpdf(y_rasp_4[n] | offsets[4] + slope * x_foldx_4[n], sigma_rasp[4]);
    }
    for (n in 1:n_mutations_per_wt_5) {
        log_lik[n_mutations_per_wt_1 + n_mutations_per_wt_2 + n_mutations_per_wt_3 + n_mutations_per_wt_4 + n] = normal_lpdf(y_rasp_5[n] | offsets[5] + slope * x_foldx_5[n], sigma_rasp[5]);
    }
    }
    """

    # Defining the data for pystan
    data = {
        "n_mutations_per_wt_1": n_mutations_per_wt[0],
        "n_mutations_per_wt_2": n_mutations_per_wt[1],
        "n_mutations_per_wt_3": n_mutations_per_wt[2],
        "n_mutations_per_wt_4": n_mutations_per_wt[3],
        "n_mutations_per_wt_5": n_mutations_per_wt[4],
        "x_foldx_1": wildtype_arrays_foldx[0],
        "x_foldx_2": wildtype_arrays_foldx[1],
        "x_foldx_3": wildtype_arrays_foldx[2],
        "x_foldx_4": wildtype_arrays_foldx[3],
        "x_foldx_5": wildtype_arrays_foldx[4],
        "y_rasp_1": wildtype_arrays_rasp[0],
        "y_rasp_2": wildtype_arrays_rasp[1],
        "y_rasp_3": wildtype_arrays_rasp[2],
        "y_rasp_4": wildtype_arrays_rasp[3],
        "y_rasp_5": wildtype_arrays_rasp[4],
    }

    # Compile the model
    stan_model = pystan.StanModel(model_code=model_code)

    # Fit the model
    fit = stan_model.sampling(data=data, iter=2000, chains=4)

    # Print results
    print(fit)

    # Save the results
    fit.to_dataframe().to_csv("comparison_foldx_rasp_pystan_results.csv")

    # Plotting the results
    # Plotting the line
    slope = fit.extract()["slope"].mean()
    print(f"Slope: {slope}")

    loo = compute_loo(fit)
    print(loo)

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
    for i, ax, x_foldx, y_rasp, pdb in zip(
        range(len(axes)), axes, wildtype_arrays_foldx, wildtype_arrays_rasp, unique_pdbs
    ):
        sns.scatterplot(x=x_foldx, y=y_rasp, ax=ax)
        ax.set_xlabel("FoldX stability")
        ax.set_ylabel("Additive RASP stability")

        intercept = fit.extract()["offsets"][:, i].mean()
        domain = np.linspace(min(x_foldx), max(x_foldx), 100)
        ax.plot(domain, slope * domain + intercept, color="red")

        ax.set_title(f"{pdb}")

    plt.show()
