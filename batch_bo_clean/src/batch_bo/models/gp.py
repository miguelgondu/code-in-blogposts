"""
A couple of gpytorch models from previous blogposts.
"""

import numpy as np
import torch
import gpytorch

from torch.distributions import Normal

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

import jax.random as jr
import gpjax as gpx
import optax as ox
from gpjax.gps import ConjugatePosterior
from botorch.fit import fit_gpytorch_mll, fit_gpytorch_mll_torch
from botorch.models import SingleTaskGP

from batch_bo.utils.constants import SEED, DEFAULT_KERNEL_GPYTORCH, LIMITS
from batch_bo.dataset import Dataset


class ExactGPScikitLearn:
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor):
        self.train_x = train_x
        self.train_y = train_y
        self._inner_gp = GaussianProcessRegressor(
            kernel=(Matern() + WhiteKernel()), n_restarts_optimizer=10
        )
        self._transform = lambda x: (x - LIMITS[0]) / (LIMITS[1] - LIMITS[0])
        self._inverse_transform = lambda x: x * (LIMITS[1] - LIMITS[0]) + LIMITS[0]
        self._y_transformer = StandardScaler()
        self._y_transformer.fit(train_y)
        self._y_transform = lambda y: self._y_transformer.transform(y)
        self._y_inverse_transform = lambda y: self._y_transformer.inverse_transform(y)
        self._y_transform = lambda y: self._y_transformer.transform(y)
        self._y_inverse_transform = lambda y: torch.from_numpy(
            self._y_transformer.inverse_transform(y.reshape(-1, 1)).reshape(-1)
        )

        print("train y: ", train_y)
        print("train y transformed: ", self._y_transform(train_y))

    def _train(self):
        self._inner_gp.fit(
            self._transform(self.train_x), self._y_transform(self.train_y)
        )

    def posterior(self, X: torch.Tensor) -> Normal:
        self._train()
        mu, std = self._inner_gp.predict(self._transform(X), return_std=True)

        return Normal(
            loc=self._y_inverse_transform(torch.tensor(mu)), scale=torch.tensor(std)
        )


class ExactGPModelJax:
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor):
        self.train_x = train_x
        self.train_y = train_y
        self.dataset = Dataset(X=train_x, y=train_y)
        self._transform = lambda x: (x - LIMITS[0]) / (LIMITS[1] - LIMITS[0])
        self._inverse_transform = lambda x: x * (LIMITS[1] - LIMITS[0]) + LIMITS[0]
        self._y_transformer = StandardScaler()
        self._y_transformer.fit(train_y)
        self._y_transform = lambda y: self._y_transformer.transform(y)
        self._y_inverse_transform = lambda y: torch.from_numpy(
            self._y_transformer.inverse_transform(y.reshape(-1, 1)).reshape(-1)
        )
        self.min_max_scaled_D = gpx.Dataset(
            X=self._transform(train_x.numpy(force=True).astype(np.float64)),
            y=self._y_transform(train_y.numpy(force=True).astype(np.float64)),
        )

        print("train y: ", train_y)
        print("train y transformed: ", self._y_transform(train_y))

        # Construct the prior
        self.meanf = gpx.mean_functions.Zero()
        self.kernel = gpx.kernels.Matern52()
        self.prior = gpx.gps.Prior(mean_function=self.meanf, kernel=self.kernel)

    def _train(self) -> ConjugatePosterior:
        # Define a likelihood
        likelihood = gpx.likelihoods.Gaussian(num_datapoints=self.min_max_scaled_D.n)

        # Construct the posterior
        posterior = self.prior * likelihood

        optimiser = ox.adam(learning_rate=1e-2)

        key = jr.key(SEED) if SEED is not None else jr.PRNGKey(np.random.randint(1))

        opt_posterior, _ = gpx.fit(
            model=posterior,
            objective=lambda p, d: -1 * gpx.objectives.conjugate_mll(p, d),
            train_data=self.min_max_scaled_D,
            optim=optimiser,
            num_iters=500,
            safe=True,
            key=key,
        )

        return opt_posterior

    def posterior(self, X: torch.Tensor) -> Normal:
        posterior = self._train()
        dist_ = posterior(
            self._transform(X.numpy(force=True).astype(np.float64)),
            self.min_max_scaled_D,
        )

        return Normal(
            loc=self._y_inverse_transform(torch.Tensor(dist_.mean().tolist())),
            scale=torch.Tensor(dist_.stddev().tolist()),
        )


class ExactGPModel(SingleTaskGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood: gpytorch.likelihoods.GaussianLikelihood = gpytorch.likelihoods.GaussianLikelihood(),
    ):
        self._transform = lambda x: (x - LIMITS[0]) / (LIMITS[1] - LIMITS[0])
        self._inverse_transform = lambda x: x * (LIMITS[1] - LIMITS[0]) + LIMITS[0]
        self._y_transformer = StandardScaler()
        self._y_transformer.fit(train_y)
        self._y_transform = lambda y: self._y_transformer.transform(y)
        self._y_inverse_transform = lambda y: torch.from_numpy(
            self._y_transformer.inverse_transform(y.reshape(-1, 1)).reshape(-1)
        )

        super(SingleTaskGP, self).__init__(
            train_inputs=self._transform(train_x),
            train_targets=torch.from_numpy(self._y_transform(train_y)).flatten(),
            likelihood=likelihood,
        )

        self.train_x = train_x
        self.train_y = train_y

        print("train y: ", train_y)
        print("train y transformed: ", self._y_transform(train_y))

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

        # try:
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        mll = fit_gpytorch_mll(mll)
        # except Exception as e:
        #     print(e)
        #     mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        #     mll = fit_gpytorch_mll(mll, closure=fit_gpytorch_mll_torch)
        self.has_been_trained = True

        assert not mll.training

        self.eval()
        return self

    def posterior(self, x: torch.Tensor) -> Normal:
        # Fit the model
        self._train()
        assert not self.training

        # Predict
        mvn = self(self._transform(x.to(torch.float64)))
        return Normal(
            loc=self._y_inverse_transform(mvn.mean.detach()),
            scale=mvn.variance.sqrt().detach(),
        )


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
