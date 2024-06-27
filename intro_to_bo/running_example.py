"""
A running example of GP regression using
GPyTorch's exact GP model.

The running example is approximating noisy samples
from f(x) = x * sin(x) in the interval [-10, 10].
"""

from pathlib import Path

import torch
from torch.distributions import Uniform, MultivariateNormal

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model

torch.manual_seed(23)

# sns.set_style("whitegrid")
# sns.set_theme(font_scale=1.5)


def f(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x) * x


domain = torch.linspace(-10, 10, 200)
codomain = f(domain)

samples: torch.Tensor = Uniform(-10, 10).sample((20,))
noise = 0.5 * torch.randn_like(f(samples))
noisy_evaluations = f(samples) + noise

ROOT_DIR = Path(__file__).parent.resolve()


def plot_with_samples(
    ax: plt.Axes, samples_: torch.Tensor = None, noisy_evaluations_: torch.Tensor = None
):
    """
    Plots and saves a simple visualization of the actual pattern
    plus the noise.
    """
    if samples_ is None:
        samples_ = samples

    if noisy_evaluations_ is None:
        noisy_evaluations_ = noisy_evaluations

    sns.lineplot(
        x=domain.detach().numpy(),
        y=codomain.detach().numpy(),
        ax=ax,
        alpha=0.5,
        linewidth=3,
        linestyle="dashed",
        c="#F75C03",
        label=r"$f(x)=x\sin(x)$",
    )

    sns.scatterplot(
        x=samples_.detach().numpy(),
        y=noisy_evaluations_.detach().numpy(),
        marker="x",
        s=125,
        c="#291720",
        ax=ax,
        label="Dataset",
    )


def plot_gp_approximation(
    ax: plt.Axes, samples_: torch.Tensor = None, noisy_evaluations_: torch.Tensor = None
) -> SingleTaskGP:
    # Fits a GP on the approximation
    # kernel = gpytorch.kernels.CosineKernel()
    mean = gpytorch.means.ZeroMean()

    if samples_ is None:
        samples_ = samples

    if noisy_evaluations_ is None:
        noisy_evaluations_ = noisy_evaluations

    gp_model = SingleTaskGP(
        samples_.unsqueeze(1), noisy_evaluations_.unsqueeze(1), mean_module=mean
    )
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
    fit_gpytorch_model(mll)
    gp_model.eval()

    prediction: MultivariateNormal = gp_model(domain)
    mean_prediction = prediction.loc.detach().numpy()
    std = torch.sqrt(prediction.variance).detach().numpy()
    lower, upper = prediction.confidence_region()

    sns.lineplot(
        x=domain.detach().numpy(),
        y=prediction.loc.detach().numpy(),
        ax=ax,
        c="b",
        linewidth=3,
        label="GP approximation w. uncertainty",
    )

    ax.fill_between(
        x=domain.detach().numpy(),
        y1=mean_prediction - std,
        y2=mean_prediction + std,
        alpha=0.3,
        color="b",
    )

    return gp_model


def plot_samples_from_gp_posterior(
    ax: plt.Axes, gp_model: SingleTaskGP, n_samples: int = 5
):
    """
    Plots samples from the GP posterior.
    """
    sns.reset_orig()
    gp_model.eval()
    domain = torch.linspace(-10, 10, 300)
    samples = gp_model(domain).sample(torch.Size([n_samples]))
    for sample in samples:
        sns.lineplot(x=domain, y=sample, ax=ax)


if __name__ == "__main__":
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))

    plot_with_samples(ax=ax)
    ax.set_ylim([-7, 9])
    gp_model = plot_gp_approximation(ax=ax)
    ax.legend(loc="upper center")

    fig_restricted_domain, ax = plt.subplots(1, 1, figsize=(9, 7 * 1))
    lower, upper = -2.5, 2.5
    mask = torch.logical_and(samples >= lower, samples <= upper)
    samples_ = samples[mask]
    noisy_evaluations_ = noisy_evaluations[mask]
    plot_with_samples(ax=ax, samples_=samples_, noisy_evaluations_=noisy_evaluations_)
    plot_gp_approximation(
        ax=ax, samples_=samples_, noisy_evaluations_=noisy_evaluations_
    )
    ax.set_ylim([-7, 9])
    ax.legend(loc="upper center")

    fig.tight_layout()
    fig.savefig(
        ROOT_DIR / "true_function_and_samples.jpg",
        dpi=120,
        bbox_inches="tight",
    )
    fig_restricted_domain.tight_layout()
    fig_restricted_domain.savefig(
        ROOT_DIR / "example_prior.jpg",
        dpi=120,
        bbox_inches="tight",
    )

    fig_samples_alone, ax = plt.subplots(1, 1, figsize=(3.75, 2.5))
    plot_samples_from_gp_posterior(ax=ax, gp_model=gp_model)
    ax.axis("off")
    fig_samples_alone.tight_layout()
    fig_samples_alone.savefig(
        ROOT_DIR / "samples_from_gp_posterior.jpg",
        dpi=80,
        bbox_inches="tight",
    )

    plt.show()
