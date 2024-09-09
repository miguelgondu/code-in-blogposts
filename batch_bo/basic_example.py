from jax import config

config.update("jax_enable_x64", True)

import gpjax as gpx
from jax import grad, jit
import jax.numpy as jnp
import jax.random as jr
import optax as ox

import matplotlib.pyplot as plt

key = jr.key(123)

f = lambda x: 10 * jnp.sin(x)

n = 50
x = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(n, 1)).sort()
y = f(x) + jr.normal(key, shape=(n, 1))
D = gpx.Dataset(X=x, y=y)

# Construct the prior
meanf = gpx.mean_functions.Zero()
kernel = gpx.kernels.RBF()
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

# Define a likelihood
likelihood = gpx.likelihoods.Gaussian(num_datapoints=n)

# Construct the posterior
posterior = prior * likelihood

# Define an optimiser
optimiser = ox.adam(learning_rate=1e-2)

# Obtain Type 2 MLEs of the hyperparameters
opt_posterior, history = gpx.fit(
    model=posterior,
    objective=lambda p, d: -1.0 * gpx.objectives.conjugate_mll(p, d),
    train_data=D,
    optim=optimiser,
    num_iters=500,
    safe=True,
    key=key,
)

# Infer the predictive posterior distribution
xtest = jnp.linspace(-3.0, 3.0, 100).reshape(-1, 1)
latent_dist = opt_posterior(xtest, D)
predictive_dist = opt_posterior.likelihood(latent_dist)

# Obtain the predictive mean and standard deviation
pred_mean = predictive_dist.mean()
pred_std = predictive_dist.stddev()

# Plot the results
plt.figure()
plt.plot(xtest, f(xtest), "r--", label="True function")
plt.plot(xtest, pred_mean, "b", label="Predictive mean")
plt.fill_between(
    xtest.flatten(),
    pred_mean.flatten() - 2 * pred_std.flatten(),
    pred_mean.flatten() + 2 * pred_std.flatten(),
    color="b",
    alpha=0.2,
    label="Predictive 95% CI",
)

plt.scatter(x, y, color="k", label="Observations")

plt.show()
