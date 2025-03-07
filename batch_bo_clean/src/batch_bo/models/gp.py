"""
A couple of gpytorch models from previous blogposts.
"""

from typing import Callable

import numpy as np
import torch
import gpytorch

from torch.distributions import Normal

from sklearn.gaussian_process import GaussianProcessRegressor

import gpjax as gpx


class ExactGPScikitLearn:
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor):
        self.train_x = train_x
        self.train_y = train_y

    def posterior(self, X: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        gp = GaussianProcessRegressor()
        gp.fit(self.train_x, self.train_y)
        mu, std = gp.predict(X, return_std=True)

        return Normal(loc=torch.tensor(mu), scale=torch.tensor(std))


class ExactGPModelJax:
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor):
        self.train_x = train_x
        self.train_y = train_y
        self.D = gpx.Dataset(X=train_x.numpy(force=True), y=train_y.numpy(force=True))

        # Construct the prior
        self.meanf = gpx.mean_functions.Zero()
        self.kernel = gpx.kernels.RBF()
        self.prior = gpx.gps.Prior(mean_function=self.meanf, kernel=self.kernel)

    def compute_posterior(self, X: torch.Tensor, y: torch.Tensor):
        # construct a dataset
        D = gpx.Dataset(X=X.numpy(force=True), y=y.numpy(force=True))

        # Define a likelihood
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)

        # Construct the posterior
        return self.prior * likelihood


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood: gpytorch.likelihoods.GaussianLikelihood = gpytorch.likelihoods.GaussianLikelihood(),
        w_ard: bool = False,
    ):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)

        self.train_x = train_x
        self.train_y = train_y

        if w_ard:
            _, n_dimensions = train_x.shape
        else:
            n_dimensions = None

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=n_dimensions)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def posterior(self, x):
        assert self.training is False
        self.eval()

        with torch.no_grad():
            return self(x)


class ExactGPModelWithLogNormalPrior(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModelWithLogNormalPrior, self).__init__(
            train_x, train_y, likelihood
        )
        _, n_dimensions = train_x.shape

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=n_dimensions,
                lengthscale_prior=gpytorch.priors.LogNormalPrior(
                    np.log(n_dimensions) / 2, 1.0
                ),
            )
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
