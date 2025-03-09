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
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
from botorch.models import SingleTaskGP

from batch_bo.utils.constants import SEED, DEFAULT_KERNEL_GPYTORCH
from batch_bo.dataset import Dataset


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
        self.dataset = Dataset(X=train_x, y=train_y)
        self.D = self.dataset.to_gpx_dataset()

        # Construct the prior
        self.meanf = gpx.mean_functions.Zero()
        self.kernel = gpx.kernels.RBF()
        self.prior = gpx.gps.Prior(mean_function=self.meanf, kernel=self.kernel)

    def _train(self) -> ConjugatePosterior:
        # Define a likelihood
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=self.D.n)

        # Construct the posterior
        posterior = self.prior * likelihood

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
            train_inputs=train_x,
            train_targets=train_y.flatten(),
            likelihood=likelihood,
        )

        self.train_x = train_x
        self.train_y = train_y

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = DEFAULT_KERNEL_GPYTORCH
        self.has_been_trained = False

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def _train(self):
        if self.has_been_trained:
            return self

        try:
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
            mll = fit_gpytorch_mll(mll)
        except Exception as e:
            print(e)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
            mll = fit_gpytorch_mll(mll, closure=fit_gpytorch_mll_torch)
        self.has_been_trained = True

        assert not mll.training

        self.eval()
        return self

    def posterior(self, x: torch.Tensor) -> Normal:
        # Fit the model
        self._train()
        assert not self.training

        # Predict
        mvn = self(x)
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
