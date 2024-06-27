import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid", font_scale=2.0)

SOLVER_NAMES = [
    "random_sampler",
    "discrete_hill_climbing",
    "one_hot_bo",
]

NAMES_BUT_PRETTY = {
    "random_sampler": "Random Sampler",
    "discrete_hill_climbing": "Discrete Hill-Climbing",
    "one_hot_bo": "Hvarfner's BO on One-Hot Encodings",
}

best_ys = {}
for solver_name in SOLVER_NAMES:
    with open(f"{solver_name}_history.json", "r") as f:
        best_y = np.array(json.load(f)["best_y"]).flatten()

    best_ys[solver_name] = best_y

fig, ax = plt.subplots(1, 1, figsize=(15, 8))

for solver_name, best_y in best_ys.items():
    best_y = np.array(best_y).flatten()
    sns.lineplot(
        x=range(len(best_y)), y=best_y, label=NAMES_BUT_PRETTY[solver_name], linewidth=3
    )

ax.set_xlabel("Iteration")
ax.set_ylabel("Best value so far")
fig.tight_layout()
fig.savefig(
    f"joint_best_y_plot_{'_'.join(SOLVER_NAMES)}.jpg", dpi=300, bbox_inches="tight"
)

plt.show()
