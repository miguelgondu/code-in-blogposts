import numpy as np
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
sns.set_theme(font_scale=1.3)


@jax.jit
def pairwise_distances(x):
    # Using the identity \|x - y\|^2 = \|x\|^2 + \|y\|^2 - 2 x^T y
    distances = (
        jnp.sum(x**2, axis=1)[:, None] - 2 * x @ x.T + jnp.sum(x**2, axis=1)[None, :]
    )

    return distances


def compute_distribution_of_distances(
    n_points: int,
    n_dimensions: int,
):
    seed = np.random.randint(0, 10000)
    key = jax.random.PRNGKey(seed)

    # Warm-up for the JIT
    x = jax.random.uniform(key, (2, n_dimensions))
    distances = pairwise_distances(x)

    # Sampling from the unit cube and computing
    # the pairwise distances
    x = jax.random.uniform(key, (n_points, n_dimensions))
    distances = pairwise_distances(x)

    # Keeping only the upper triangular part
    # of the distance matrix
    distances = jnp.triu(distances, k=1)

    return distances[distances > 0.0]


if __name__ == "__main__":
    N_POINTS = 1_000
    n_dimensions = [2**exp_ for exp_ in range(1, 8)]

    arrays = {}
    means = {}
    for n_dimensions_ in n_dimensions:
        distances = compute_distribution_of_distances(N_POINTS, n_dimensions_)
        arrays[n_dimensions_] = distances
        means[n_dimensions_] = jnp.mean(distances)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for n_dimensions_ in n_dimensions:
        sns.histplot(arrays[n_dimensions_], kde=True, label=f"{n_dimensions_}", ax=ax)

    ax.set_title("Distribution of pairwise distances - Sampled from unit cube")
    ax.set_xlabel("Pairwise distances")
    ax.set_ylabel("Density")

    fig.legend()
    fig.tight_layout()

    fig.savefig("pairwise_distances_unit_normal.jpg")
    print([(np.sqrt(n_dimension), means[n_dimension]) for n_dimension in n_dimensions])
    plt.show()
