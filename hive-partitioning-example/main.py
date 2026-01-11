from synthetic_csv import run_experiment as run_experiment_csv
from synthetic_pydantic import run_experiment as run_experiment_json
from analysis import load_results_csv, load_results_json


if __name__ == "__main__":
    proteins = [
        "9L76",
        "9M3K",
        "9LYT",
        "9LJC",
        "9ODL",
    ]
    molecules = [
        "mol1",
        "mol2",
        "mol3",
        "mol4",
    ]
    run_experiment_csv(proteins=proteins, molecules=molecules, repetitions=10)
    run_experiment_json(proteins=proteins, molecules=molecules, repetitions=10)

    df_from_csvs = load_results_csv()
    df_from_json = load_results_json()

    print(df_from_csvs)
    print(df_from_json)
