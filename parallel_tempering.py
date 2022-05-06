from functools import partial
from time import perf_counter
from tqdm import tqdm

import numpy as np
from numpy.random import normal

from utils import d_fold_tensor_product, sample_sphere, get_normal_proposal


class ParallelTempering:
    def __init__(
        self,
        lmbda,
        dim=10,
        order=4,
        cycles=1_000,
        cycle_length=100,
        warmup_cycles=50,
        warmup_cycle_length=1000,
        betas=[0.05 * i for i in range(1, 11)],
        get_proposal=get_normal_proposal,
        verbose=False,
        store_chain=False,
        seed=False,
    ) -> None:

        # Signal-to-noise parameter
        self.lmbda = lmbda

        # Dimension
        self.dim = dim

        # Tensor order
        self.order = order

        # Number of cycles for MCMC iteration
        self.cycles = cycles
        self.cycle_length = cycle_length

        # Number of warmup cycles
        self.warmup_cycles = warmup_cycles
        self.warmup_cycle_length = warmup_cycle_length

        # Inverse temperatures
        self.betas = betas

        # Fix random seed
        if seed:
            self.seed = seed

        self.verbose = verbose

        self.generate_sample()

        self.get_proposal = get_proposal

        # estimated spike (mean)
        self.estimate = np.zeros(self.dim)

        self.store_chain = store_chain
        if store_chain:
            # prepare storage for sample chain
            self.chain = np.zeros((self.cycles, self.dim))

        # inner products of the spike and the estimated spikes, updated after each cycle
        self.correlations = np.zeros(self.cycles + 1)

        # the acceptance rate of the chain with beta=1 over all sampling steps
        self.acceptance_rate = 0

        # how long self.run_PT took
        self.runtime = None

    def log_posterior(self, x) -> float:
        """log-posterior density in the model with uniform prior on the sphere
        and asymmetric Gaussian noise. This ignores terms constant wrt x,
        since they are irrelevant for the MH steps/replica swaps."""
        n, d = self.dim, self.order

        # this loop computes < y, x^{\otimes d} >
        correlation = self.sample
        for _ in range(d):
            correlation = correlation @ x

        return n * self.lmbda * correlation

    def generate_sample(self) -> None:
        n, d = self.dim, self.order

        self.spike = sample_sphere(n)  # sampled uniformly from the sphere
        W = normal(0, 1, d * (n,))  # Gaussian noise

        # Y = lambda*x^{\otimes d} + W/sqrt{n}
        self.sample = self.lmbda * d_fold_tensor_product(self.spike, d) + W / np.sqrt(n)

    def get_update_factor(self, acceptance_rate) -> float:
        """Returns a factor to update the scaling parameters
        depending on the acceptance rate."""
        factor = 1.0
        if acceptance_rate < 0.02:
            factor = 1.5
        elif 0.02 <= acceptance_rate < 0.05:
            factor = 1.3
        elif 0.05 <= acceptance_rate < 0.10:
            factor = 1.2
        elif 0.10 <= acceptance_rate < 0.15:
            factor = 1.05
        elif 0.35 <= acceptance_rate < 0.50:
            factor = 0.95
        elif 0.5 <= acceptance_rate:
            factor = 0.8

        return factor

    def replica_swaps(self) -> None:
        parity = np.random.choice([0, 1])
        # decide to swap replica i and i+1 for even or odd i. 0 corresponds to even, 1 to odd

        for smaller_beta, bigger_beta in zip(
            self.betas[parity::2], self.betas[(parity + 1) :: 2]
        ):

            # compute acceptance probability
            log_acceptance_probability = (bigger_beta - smaller_beta) * (
                self.log_posterior(self.current_state[bigger_beta])
                - self.log_posterior(self.current_state[smaller_beta])
            )

            if (
                0 < log_acceptance_probability
                or -np.random.exponential() < log_acceptance_probability
            ):
                # equivalent to np.log(np.random.uniform()) < log_acceptance_probability
                self.total_swaps += 1
                # swap states i and i+1
                self.current_state[smaller_beta], self.current_state[bigger_beta] = (
                    self.current_state[bigger_beta],
                    self.current_state[smaller_beta],
                )

    def mh_step(self, x_old, beta=1) -> list[np.ndarray, int]:
        """Takes one Metropolis Hastings step. Assumes proposal density is symmetric"""
        proposal = self.get_proposal(x_old, self.scaling_parameters[beta])
        r = beta * (self.log_posterior(proposal) - self.log_posterior(x_old))

        if (
            0 < r or -np.random.exponential() < r
        ):  # equivalent to np.log(np.random.uniform()) < r
            return proposal, 1
        else:
            return x_old, 0

    def run_cycle(self, cycle_length=100, beta=1) -> list[np.ndarray, int]:
        """Takes cycle_length many Metropolis Hastings steps"""
        acceptance_rate = 0
        x = self.current_state[beta]
        for _ in range(cycle_length):
            x, accepted = self.mh_step(x, beta)
            acceptance_rate += accepted

        acceptance_rate /= cycle_length

        return x, acceptance_rate

    def run_PT(self) -> None:
        start_time = perf_counter()

        # initialize states for all temperatures
        # this is a dict which maps beta -> current state
        self.current_state = {beta: sample_sphere(self.dim) for beta in self.betas}

        # initial correlation
        self.correlations[0] = self.current_state[1] @ self.spike

        # storage for scaling parameters (variance) of jumping distribution
        # this is a dict which maps beta -> scaling parameter
        self.scaling_parameters = dict.fromkeys(self.betas, 1.0)

        if self.verbose:
            print(f"[lambda={self.lmbda:.1f}, dim={self.dim}] Starting warmup cycles.")

        # run warmup cycles for all temperatures
        for beta in self.betas:
            cycle_iterator = (
                tqdm(
                    range(self.warmup_cycles),
                    desc=f"[lambda={self.lmbda:.1f}, dim={self.dim}] beta={beta:.1f} WARMUP",
                )
                if self.verbose
                else range(self.cycles)
            )
            for i in cycle_iterator:
                self.current_state[beta], acceptance_rate = self.run_cycle(
                    self.warmup_cycle_length, beta
                )

                # adapt acceptance rate
                factor = self.get_update_factor(acceptance_rate)
                self.scaling_parameters[beta] *= factor

            if self.verbose:
                print(
                    f"[lambda={self.lmbda:.1f}, dim={self.dim}] Finished warmup cycles for beta={beta:.1f}. Final acceptance rate was {int(100*acceptance_rate)}%."
                )

        if self.verbose:
            print(
                f"[lambda={self.lmbda:.1f}, dim={self.dim}] Finished warmup. Starting sampling."
            )

        # run sampling cycles
        self.acceptance_rate = 0
        self.total_swaps = 0

        cycle_iter = (
            tqdm(
                range(
                    1, self.cycles + 1
                ),  # shift by one for easier online computation of acceptance rates
                desc=f"[lambda={self.lmbda:.1f}, dim={self.dim}] SAMPLING",
            )
            if self.verbose
            else range(1, self.cycles)
        )

        for i in cycle_iter:
            for beta in self.betas:
                self.current_state[beta], acceptance_rate = self.run_cycle(
                    self.cycle_length, beta
                )

            # update acceptance rate (for beta=1)
            self.acceptance_rate *= i - 1
            self.acceptance_rate += acceptance_rate
            self.acceptance_rate /= i

            self.replica_swaps()

            if self.store_chain:
                self.chain[i] = self.current_state[1]

            # update estimate
            self.estimate *= i
            self.estimate += self.current_state[1]
            self.estimate /= i + 1

            self.correlations[i] = self.estimate @ self.spike

            if self.verbose and ((i + 1) % (self.cycles // 10) == 0):
                print(
                    f"[lambda={self.lmbda:.1f}, dim={self.dim}] Finished {i+1} cycles. Current correlation is {self.correlations[i]}. Acceptance rate so far is {int(100*self.acceptance_rate)}%."
                )

        end_time = perf_counter()
        self.runtime = end_time - start_time

        if self.verbose:
            print(
                f"[lambda={self.lmbda:.1f}, dim={self.dim}] Finished sampling. Correlation was {self.correlations[-1]:.2f}. Final acceptance rate was {int(100*self.acceptance_rate)}%. There were {self.total_swaps} swaps. Runtime was {self.runtime:.0f}s."
            )

        print(f"[lambda={self.lmbda:.1f}, dim={self.dim}] Done.")
