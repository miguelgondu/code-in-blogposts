"""
A couple of gpytorch models from previous blogposts.
"""

import numpy as np
import torch
import gpytorch

from torch.distributions import Normal

from sklearn.gaussian_process import GaussianProcessRegressor

import jax.random as jr
import gpjax as gpx
import optax as ox
from gpjax.gps import ConjugatePosterior
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP

from batch_bo.utils.constants import SEED, LIMITS, DEFAULT_KERNEL_GPYTORCH


class ExactGPScikitLearn:
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor):
        self.train_x = train_x
        self.train_y = train_y
        self._inner_gp = GaussianProcessRegressor()

    def _train(self):
        self._inner_gp.fit(self.train_x, self.train_y)

    def posterior(self, X: torch.Tensor) -> Normal:
        self._train()
        mu, std = self._inner_gp.predict(X, return_std=True)

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

    def _train(self) -> ConjugatePosterior:
        posterior = self.compute_posterior(self.train_x, self.train_y)
        optimiser = ox.adam(learning_rate=1e-2)

        key = jr.key(SEED)

        opt_posterior, _ = gpx.fit(
            model=posterior,
            objective=lambda p, d: -1 * gpx.objectives.conjugate_mll(p, d),
            train_data=self.D,
            optim=optimiser,
            num_iters=500,
            safe=True,
            key=key,
        )

        return opt_posterior

    def posterior(self, X: torch.Tensor) -> Normal:
        posterior = self._train()
        dist_ = posterior(X.numpy(force=True).astype(np.float64), self.D)

        return Normal(
            loc=torch.Tensor(dist_.mean().tolist()),
            scale=torch.Tensor(dist_.stddev().tolist()),
        )


class ExactGPModel(SingleTaskGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood: gpytorch.likelihoods.GaussianLikelihood = gpytorch.likelihoods.GaussianLikelihood(),
    ):
        super(SingleTaskGP, self).__init__(
            train_inputs=(train_x - LIMITS[0]) / (LIMITS[1] - LIMITS[0]),
            train_targets=train_y.flatten(),
            likelihood=likelihood,
        )

        self.train_x = train_x
        self.train_y = train_y

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = DEFAULT_KERNEL_GPYTORCH

    def forward(self, x):
        min_max_x = (x - LIMITS[0]) / (LIMITS[1] - LIMITS[0])
        mean_x = self.mean_module(min_max_x)
        covar_x = self.covar_module(min_max_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def _train(self):
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        mll = fit_gpytorch_mll(mll)
        assert not mll.training

        self.eval()
        return self

    def posterior(self, x: torch.Tensor) -> Normal:
        # Fit the model
        self._train()
        assert not self.training

        # Predict
        min_max_x = (x - LIMITS[0]) / (LIMITS[1] - LIMITS[0])
        mvn = self(min_max_x)
        return Normal(loc=mvn.mean.detach(), scale=mvn.variance.sqrt().detach())


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

    def _train(self):
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        mll = fit_gpytorch_mll(mll)
        assert not mll.training

        self.eval()
        return self

    def posterior(self, x) -> Normal:
        # Fit the model
        self._train()
        assert not self.training

        # Predict
        mvn = self(x)
        return Normal(loc=mvn.mean, scale=mvn.variance.sqrt())
