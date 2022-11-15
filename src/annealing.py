import jax.numpy as np
from jax import random
from tqdm import tqdm

from utils import get_normal_proposal, sample_sphere


class SimulatedAnnealing:
    """
    Implementation of simulated annealing for tensor PCA.
    """

    def __init__(
        self,
        log_posterior,
        spike,
        Y,
        dim,
        lmbda,
        key,
        initial_temperature=2,
        cooling_rate=0.95,
        order=4,
        get_proposal=get_normal_proposal,
        n_steps=2000,
        verbose=False,
    ) -> None:
        self.log_posterior = log_posterior
        self.spike = spike
        self.Y = Y
        self.dim = dim
        self.lmbda = lmbda
        self.key = key
        self.temperature = self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate

        # Tensor order.
        self.order = order

        # Proposal sampler.
        self.get_proposal = get_proposal

        # Number of steps to take.
        self.n_steps = n_steps

        # Initialize starting state.
        self.key, subkey = random.split(self.key)
        self.current_state = sample_sphere(subkey, dim)

        self.verbose = verbose

    def _temperature_scheduler(self, n) -> float:
        """Returns temperature depending on number of iteration using
        an exponential cooling schedule."""
        temperature = self.initial_temperature * self.cooling_rate**n 
        """
        Antoine: I think that e.g. in https://arxiv.org/pdf/2206.04760.pdf they use a linear cooling schedule (cf second column of page 4).
        We could compare the two, my fear is that an exponential schedule will lead to too fast cooling, even more if you don't do a full cycle at each decrease 
        of temperature (cf below).
        """
        return temperature

    def _step(self, key, x_old, T) -> tuple[np.DeviceArray, int]:
        key, subkey = random.split(key)
        proposal = self.get_proposal(subkey, x_old)
        r = (
            self.log_posterior(proposal, self.Y) - self.log_posterior(x_old, self.Y)
        ) / T

        # Equivalent to np.random.uniform() < exp(r).
        if -random.exponential(key) < r:
            return proposal, 1
        else:
            return x_old, 0

    """
    Antoine: If you look at usual SA algorithms (like in https://arxiv.org/pdf/2206.04760.pdf), after a decrease of temperature, 
    they do a ``Monte Carlo Sweep'', that is they change every variable in a model with discrete variables: I think here this corresponds to do a full MC cycle rather than 
    just a single step after changing the temperature! The idea of SA is to equilibrate at each intermediate temperature, so when we go to T - dT
    we hope to have equilibrated at T already.
    """
    def run(self) -> None:
        n_steps = tqdm(range(self.n_steps)) if self.verbose else range(self.n_steps)
        for n in n_steps:
            temperature = self._temperature_scheduler(n)
            self.key, subkey = random.split(self.key)
            self.current_state = self._step(subkey, self.current_state, temperature)


class ReplicatedAnnealing:
    """
    Implementation of replicated simlated annealing for tensor PCA.
    """

    def __init__(self) -> None:
        raise NotImplementedError
