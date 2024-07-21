# This is my quick and hacky implementation
# of Expected Improvement. It is based on
# my implementation of PI in GPJax. I include
# the copyright notice from GPJax just
# because I'm using their code as a reference.

# Copyright 2024 The JaxGaussianProcesses Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from dataclasses import dataclass

from beartype.typing import Mapping
from jaxtyping import Num
import tensorflow_probability.substrates.jax as tfp

from gpjax.dataset import Dataset
from gpjax.decision_making.utility_functions.base import (
    AbstractSinglePointUtilityFunctionBuilder,
    SinglePointUtilityFunction,
)
from gpjax.decision_making.utils import OBJECTIVE
from gpjax.gps import ConjugatePosterior
from gpjax.typing import (
    Array,
    KeyArray,
)


@dataclass
class ExpectedImprovement(AbstractSinglePointUtilityFunctionBuilder):
    def build_utility_function(
        self,
        posteriors: Mapping[str, ConjugatePosterior],
        datasets: Mapping[str, Dataset],
        key: KeyArray,
    ) -> SinglePointUtilityFunction:
        """
        Constructs the probability of improvement utility function
        using the predictive posterior of the objective function.

        Args:
            posteriors (Mapping[str, AbstractPosterior]): Dictionary of posteriors to be
            used to form the utility function. One of the posteriors must correspond
            to the `OBJECTIVE` key, as we sample from the objective posterior to form
            the utility function.
            datasets (Mapping[str, Dataset]): Dictionary of datasets which may be used
            to form the utility function. Keys in `datasets` should correspond to
            keys in `posteriors`. One of the datasets must correspond
            to the `OBJECTIVE` key.
            key (KeyArray): JAX PRNG key used for random number generation. Since
            the probability of improvement is computed deterministically
            from the predictive posterior, the key is not used.

        Returns:
            SinglePointUtilityFunction: the probability of improvement utility function.
        """
        self.check_objective_present(posteriors, datasets)

        objective_posterior = posteriors[OBJECTIVE]
        if not isinstance(objective_posterior, ConjugatePosterior):
            raise ValueError(
                "Objective posterior must be a ConjugatePosterior to compute the Probability of Improvement using a Gaussian CDF."
            )

        objective_dataset = datasets[OBJECTIVE]
        if (
            objective_dataset.X is None
            or objective_dataset.n == 0
            or objective_dataset.y is None
        ):
            raise ValueError(
                "Objective dataset must be non-empty to compute the "
                "Probability of Improvement (since we need a "
                "`best_y` value)."
            )

        def expected_improvement(x_test: Num[Array, "N D"]):
            # Computing the posterior mean for the training dataset
            # for computing the best_y value (as the minimum
            # posterior mean of the objective function)
            predictive_dist_for_training = objective_posterior.predict(
                objective_dataset.X, objective_dataset
            )
            best_y = predictive_dist_for_training.mean().min()

            predictive_dist = objective_posterior.predict(x_test, objective_dataset)

            normal_dist = tfp.distributions.Normal(
                loc=predictive_dist.mean(),
                scale=predictive_dist.stddev(),
            )

            ei = (best_y - predictive_dist.mean()) * normal_dist.cdf(
                best_y
            ) + predictive_dist.stddev() * normal_dist.prob(best_y)

            return ei.reshape(-1, 1)

        return expected_improvement
