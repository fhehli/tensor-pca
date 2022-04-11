from functools import partial
from time import time

import numpy as np
from numpy.random import normal

from utils import tensorize, normalise, sample_sphere, get_normal_proposal


class SpikedTensor:
    def __init__(
        self,
        lmbda,
        dim=10,
        order=4,
        cycles=1_000,
        warmup_cycles=100,
        cycle_length=100,
        betas=[0.05 * i for i in range(1, 21)],
        get_proposal=get_normal_proposal,
        store_chain=False,
        seed=False,
    ) -> None:

        # Signal-to-noise ratio
        self.lmbda = lmbda

        # Dimension
        self.dim = dim

        # Tensor order
        self.order = order

        # Number of cycles for MCMC iteration
        self.cycles = cycles

        # Number of warmup cycles
        self.warmup_cycles = warmup_cycles

        self.cycle_length = cycle_length

        # Inverse temperatures
        self.betas = betas

        # Fix random seed
        if seed:
            self.seed = seed

        self.generate_sample()

        self.get_proposal = get_proposal

        # estimated spike (mean)
        self.estimate = np.zeros(self.dim)

        self.store_chain = store_chain
        if store_chain:
            # prepare storage for sample chain
            self.chain = np.zeros((self.cycles, self.dim))

        # inner product of the spike and the estimated spike
        self.correlation = None

        self.runtime = None

    def generate_sample(self) -> None:
        n, d = self.dim, self.order

        self.spike = sample_sphere(n)  # sampled uniformly from the sphere
        W = normal(0, 1, d * (n,))  # gaussian noise

        # Y = lambda*x^{\otimes d} + W/sqrt{n}
        self.sample = self.lmbda * tensorize(self.spike, d) + W / np.sqrt(n)

        # posterior density for beta=1
        self.log_posterior = (
            lambda x: -n / 2 * np.sum((self.sample - self.lmbda * tensorize(x, d)) ** 2)
        )

        # posterior densities for all temperatures
        self.log_posteriors = {
            beta: lambda x: self.log_posterior(x) ** beta for beta in self.betas
        }

    @staticmethod
    def mh_step(x_old, log_posterior, get_proposal) -> list[np.ndarray, int]:
        # assumes proposal density is symmetric
        proposal = get_proposal(x_old)
        r = log_posterior(proposal) - log_posterior(x_old)

        if np.log(np.random.uniform()) < r:
            return proposal, 1
        else:
            return x_old, 0

    def run_cycle(self, beta=1) -> list[np.ndarray, int]:
        acceptance_rate = 0
        for _ in range(self.cycle_length):
            x_new, accepted = self.mh_step(
                self.current_states[beta],
                self.log_posteriors[beta],
                partial(
                    self.get_proposal, scaling_parameter=self.scaling_parameters[beta]
                ),
            )
            acceptance_rate += accepted

        acceptance_rate /= self.cycle_length

        return x_new, acceptance_rate

    def replica_swaps(self):
        parity = np.random.choice([0, 1])
        for smaller_beta, bigger_beta in zip(
            self.betas[parity::2], self.betas[(parity + 1) :: 2]
        ):
            # compute acceptance probability
            log_acceptance_probability = (bigger_beta - smaller_beta) * (
                self.log_posterior(self.current_states[bigger_beta])
                - self.log_posterior(self.current_states[smaller_beta])
            )
            if np.log(np.random.uniform()) < log_acceptance_probability:
                # swap states i and i+1
                self.current_states[smaller_beta], self.current_states[bigger_beta] = (
                    self.current_states[bigger_beta],
                    self.current_states[smaller_beta],
                )

    def run_PT(self) -> list[list, float]:
        # initialize states for all temperatures
        self.current_states = {beta: sample_sphere(self.dim) for beta in self.betas}

        # storage for scaling parameters (variance) of jumping distribution
        self.scaling_parameters = dict.fromkeys(self.betas, 1.0)

        start = time()

        # run warmup cycles for all temperatures
        for beta in self.betas:
            for _ in range(self.warmup_cycles):
                _, acceptance_rate = self.run_cycle(beta)

                # tune acceptance rate
                if acceptance_rate < 0.20:
                    self.scaling_parameters[beta] *= 1.1
                elif acceptance_rate > 0.40:
                    self.scaling_parameters[beta] *= 0.9

        # run cycles
        for i in range(self.cycles):
            for beta in self.betas:
                self.current_states[beta], _ = self.run_cycle(beta)

            self.replica_swaps()
            self.estimate += self.current_states[1]
            if self.store_chain:
                self.chain[i] = self.current_states[1]

        self.estimate /= self.cycles
        self.correlation = self.estimate @ self.spike

        end = time()
        self.runtime = end - start
