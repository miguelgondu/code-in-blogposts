import duckdb


def load_results_csv():
    local_connection = duckdb.connect()

    query = """
    SELECT protein, molecule, seed, binding_affinity, accuracy
    FROM read_csv("results/**/*.csv", hive_partitioning=true)
    """

    return local_connection.execute(query).df()


def load_results_json():
    local_connection = duckdb.connect()

    query = """
    SELECT protein, molecule, seed, binding_affinity, accuracy, metadata.model_used
    FROM read_json("results/**/*.json", hive_partitioning=true)
    """

    return local_connection.execute(query).df()
