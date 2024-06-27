import math
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

# from scipy.stats import gamma

sns.set_style("darkgrid")
sns.set_theme(font_scale=1.3)


def plot_gamma_density():
    lengthscale_domain = np.linspace(0.01, 2.0, 1000)
    loc = 3
    rate = 6

    density = (
        (rate**loc / (math.factorial(loc - 1)))
        * lengthscale_domain ** (loc - 1)
        * np.exp(-lengthscale_domain * rate)
    )
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    sns.lineplot(x=lengthscale_domain, y=density, ax=ax)
    ax.set_title("Density of a Gamma(loc=3, rate=6) distribution")
    ax.set_xlabel("Lengthscale")
    ax.set_ylabel("Density")

    fig.savefig("gamma_density_.jpg", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    plot_gamma_density()
    plt.show()
