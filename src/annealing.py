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
        cooling_rate=1e-3,
        order=4,
        get_proposal=get_normal_proposal,
        n_steps=1990,
        cycle_length=100,
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

        self.cycle_length = cycle_length

        # Initialize starting state.
        self.key, subkey = random.split(self.key)
        self.current_state = sample_sphere(subkey, dim)

        self.verbose = verbose

    def _temperature_scheduler(self, n) -> float:
        """Returns temperature depending on number of iteration using
        a linear cooling schedule. By default, the initial temperature
        is 2, the cooling rate is 1/1000 and the number of "sweeps" is
        1990. Hence, the temperature decreases from 2 down to 0.01."""
        # TODO: Try exponential cooling schedule, i.e.
        # temperature = self.initial_temperature * self.cooling_rate**n
        temperature = self.initial_temperature - self.cooling_rate * n
        return temperature

    def _step(self, key, x_old, temperature) -> np.DeviceArray:
        key, subkey = random.split(key)
        proposal = self.get_proposal(subkey, x_old)
        r = (
            self.log_posterior(proposal, self.Y) - self.log_posterior(x_old, self.Y)
        ) / temperature

        # Equivalent to np.random.uniform() < exp(r).
        if -random.exponential(key) < r:
            return proposal
        else:
            return x_old

    def _run_cycle(self, key, temperature) -> np.DeviceArray:
        """Takes cycle_length many Metropolis steps."""
        x = self.current_state
        for _ in range(self.cycle_length):
            key, subkey = random.split(key)
            x = self._mh_step(subkey, x, temperature)
        return x

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
            self.current_state = self._run_cycle(subkey, temperature)


class ReplicatedAnnealing:
    """
    Implementation of replicated simlated annealing for tensor PCA.
    """

    def __init__(self) -> None:
        raise NotImplementedError
