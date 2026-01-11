from synthetic import run_experiment
from analysis import load_results


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
    run_experiment(proteins=proteins, molecules=molecules, repetitions=10)

    df = load_results()

    print(df)
