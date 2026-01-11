import duckdb


def load_results():
    local_connection = duckdb.connect()

    query = """
    SELECT protein, molecule, seed, binding_affinity, accuracy
    FROM read_csv("results/**/*.csv", hive_partitioning=true)
    """

    return local_connection.execute(query).df()


if __name__ == "__main__":
    load_results()
