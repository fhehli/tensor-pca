from time import perf_counter
from tqdm import tqdm
from functools import partial

import numpy as np
from numpy.random import normal

from utils import d_fold_tensor_product, sample_sphere, get_normal_proposal


class SpikedTensor:
    def __init__(
        self,
        lmbda,
        dim,
        order=4,
        seed=None,
    ) -> None:

        # Signal-to-noise parameter
        self.lmbda = lmbda

        # Dimension
        self.dim = dim

        # Tensor order
        self.order = order

        # Fix random seed
        if seed is not None:
            np.random.seed(seed)

        # generate a sample
        self.spike, self.sample = SpikedTensor.generate_sample(
            self.lmbda, self.dim, self.order
        )

    @staticmethod
    def generate_sample(lmbda, n, d) -> None:
        """Generates a sample of the form Y = lambda*x^{\otimes d} + W/sqrt(n)"""
        spike = sample_sphere(n)  # sampled uniformly from the sphere
        W = normal(0, 1, d * (n,))  # iid standard normal noise
        sample = lmbda * d_fold_tensor_product(spike, d) + W / np.sqrt(n)

        return spike, sample


class ParallelTempering(SpikedTensor):
    def __init__(
        self,
        log_posterior,
        lmbda,
        dim,
        order=4,
        cycles=1_000,
        cycle_length=100,
        warmup_cycles=50,
        warmup_cycle_length=1000,
        betas=[0.05 * i for i in range(1, 11)],
        swap_frequency=1,
        get_proposal=get_normal_proposal,
        tol=5e-3,
        tol_window=20,
        verbose=False,
        store_chain=False,
        seed=None,
    ) -> None:

        super().__init__(lmbda, dim, order, seed)

        # Log posterior density
        self.log_posterior = partial(
            log_posterior,
            Y=self.sample,
            lmbda=lmbda,
            dim=dim,
        )

        # Proposal sampler
        self.get_proposal = get_proposal

        # Number of cycles for MCMC iteration
        self.cycles = cycles
        self.cycle_length = cycle_length

        # Number of warmup cycles
        self.warmup_cycles = warmup_cycles
        self.warmup_cycle_length = warmup_cycle_length

        # Inverse temperatures
        self.betas = betas

        # How frequently to perform replica swaps
        self.swap_frequency = swap_frequency

        # Stopping tolerance
        self.tol = tol
        self.tol_window = tol_window

        self.verbose = verbose
        self.store_chain = store_chain

        # The acceptance rate of the chain with beta=1 over all sampling steps
        self.acceptance_rate = 0

        # How long self.run_PT took
        self.runtime = None

        # Storage for scaling parameters (variance) of jumping distribution
        # This is a dict which maps beta -> scaling parameter
        self.scaling_parameters = dict.fromkeys(self.betas, 1.0)

        # Estimated spike (mean)
        self.estimate = np.zeros(self.dim)

        # Initialize states for all temperatures
        # This is a dict which maps beta -> current state
        self.current_state = {beta: sample_sphere(self.dim) for beta in self.betas}

        if store_chain:
            # Prepare storage for sample chain
            self.chain = np.zeros((self.cycles, self.dim))
            self.chain[0] = self.current_state[1]

        # Inner products of the spike and the estimated spikes, updated after each cycle
        self.correlations = np.zeros(self.cycles + 1)
        self.correlations[0] = self.current_state[1] @ self.spike

    def get_update_factor(self, acceptance_rate) -> float:
        """Returns a factor to update the scaling parameters
        depending on the acceptance rate. We aim for an acceptance
        rate between 20% and 30%."""
        factor = 1.0
        if acceptance_rate < 0.02:
            factor = 1.5
        elif 0.02 <= acceptance_rate < 0.05:
            factor = 1.3
        elif 0.05 <= acceptance_rate < 0.10:
            factor = 1.2
        elif 0.10 <= acceptance_rate < 0.15:
            factor = 1.05
        elif 0.15 <= acceptance_rate < 0.20:
            factor = 1.02
        elif 0.30 <= acceptance_rate < 0.35:
            factor = 0.98
        elif 0.35 <= acceptance_rate < 0.50:
            factor = 0.95
        elif 0.5 <= acceptance_rate:
            factor = 0.8

        return factor

    def replica_swaps(self) -> None:
        # Decide to swap replica i and i+1 for even or odd i:
        # parity=0 corresponds to even, parity=1 to odd
        parity = np.random.choice([0, 1])

        for smaller_beta, bigger_beta in zip(
            self.betas[parity::2], self.betas[(parity + 1) :: 2]
        ):
            # (log) acceptance probability
            r = (bigger_beta - smaller_beta) * (
                self.log_posterior(self.current_state[bigger_beta])
                - self.log_posterior(self.current_state[smaller_beta])
            )

            # Equivalent to np.log(np.random.uniform()) < r.
            if 0 < r or -np.random.exponential() < r:
                self.total_swaps += 1
                # Swap states i and i+1.
                self.current_state[smaller_beta], self.current_state[bigger_beta] = (
                    self.current_state[bigger_beta],
                    self.current_state[smaller_beta],
                )

    def mh_step(self, x_old, beta=1) -> list[np.ndarray, int]:
        """Takes one Metropolis step. Assumes proposal density is symmetric."""
        proposal = self.get_proposal(x_old, self.scaling_parameters[beta])
        r = beta * (self.log_posterior(proposal) - self.log_posterior(x_old))

        # Equivalent to np.log(np.random.uniform()) < r.
        if 0 < r or -np.random.exponential() < r:
            return proposal, 1
        else:
            return x_old, 0

    def run_cycle(self, cycle_length=100, beta=1) -> list[np.ndarray, int]:
        """Takes cycle_length many Metropolis steps.

        Returns new state and acceptance rate."""
        n_accepted = 0
        x = self.current_state[beta]
        for _ in range(cycle_length):
            x, accepted = self.mh_step(x, beta)
            n_accepted += accepted

        return x, n_accepted / cycle_length

    def run_PT(self) -> None:
        start_time = perf_counter()

        if self.verbose:
            print(f"[lambda={self.lmbda:.1f}, dim={self.dim}] Starting warmup cycles.")

        # Warmup cycles for all temperatures.
        for beta in self.betas:
            # Define an iterator with progress bar if in verbose mode.
            cycle_iterator = (
                tqdm(
                    range(self.warmup_cycles),
                    desc=f"[lambda={self.lmbda:.1f}, dim={self.dim}] beta={beta:.2f} WARMUP",
                )
                if self.verbose
                else range(self.cycles)
            )
            # Run cycles
            for i in cycle_iterator:
                self.current_state[beta], acceptance_rate = self.run_cycle(
                    self.warmup_cycle_length, beta
                )

                # Update scaling to optimize acceptance rate.
                factor = self.get_update_factor(acceptance_rate)
                self.scaling_parameters[beta] *= factor

            if self.verbose:
                print(
                    f"[lambda={self.lmbda:.1f}, dim={self.dim}] Finished warmup cycles for beta={beta:.2f}. Final acceptance rate was {int(100*acceptance_rate)}%."
                )

        if self.verbose:
            print(
                f"[lambda={self.lmbda:.1f}, dim={self.dim}] Finished warmup. Starting sampling."
            )

        ## SAMPLING ##
        self.acceptance_rate = 0
        self.total_swaps = 0

        # Define an iterator with progress bar if in verbose mode.
        # Shifted by one for easier online computation of acceptance rates
        cycle_iter = (
            tqdm(
                range(1, self.cycles + 1),
                desc=f"[lambda={self.lmbda:.1f}, dim={self.dim}] SAMPLING",
            )
            if self.verbose
            else range(1, self.cycles + 1)
        )

        # Run cycles.
        for i in cycle_iter:
            for beta in self.betas:
                self.current_state[beta], acceptance_rate = self.run_cycle(
                    self.cycle_length, beta
                )

            # Update acceptance rate (for beta=1).
            self.acceptance_rate *= i - 1
            self.acceptance_rate += acceptance_rate
            self.acceptance_rate /= i

            if (i + 1) % self.swap_frequency == 0:
                self.replica_swaps()

            if self.store_chain:
                self.chain[i] = self.current_state[1]

            # Update estimated spike.
            self.estimate *= i
            self.estimate += self.current_state[1]
            self.estimate /= i + 1

            self.correlations[i] = self.estimate @ self.spike

            # Check "convergence":
            # We check whether the correlation of the last n samples
            # lies within an interval of size 2*self.tol, where n = self.tol_window.
            if i + 1 >= self.tol_window and np.alltrue(
                np.abs(
                    self.correlations[i - self.tol_window + 1 : i + 1]
                    - self.correlations[i - self.tol_window + 1 : i + 1].mean()
                )
                < self.tol
            ):
                if self.verbose:
                    print(
                        f"[lambda={self.lmbda:.1f}, dim={self.dim}] Correlation has converged after {i} cycles."
                    )
                # Omit non-measured correlations.
                self.correlations = self.correlations[:i]
                break  # Stop sampling.

            if self.verbose and ((i + 1) % (self.cycles // 10) == 0):
                print(
                    f"[lambda={self.lmbda:.1f}, dim={self.dim}] Finished {i+1} cycles. Current correlation is {self.correlations[i]:.2f}. Acceptance rate so far is {int(100*self.acceptance_rate)}%."
                )

        end_time = perf_counter()
        self.runtime = end_time - start_time

        if self.verbose:
            print(
                f"[lambda={self.lmbda:.1f}, dim={self.dim}] Finished sampling. Correlation was {self.correlations[-1]:.2f}. Final acceptance rate was {int(100*self.acceptance_rate)}%. There were {self.total_swaps} swaps. Runtime was {self.runtime:.0f}s."
            )


if __name__ == "__main__":

    def log_posterior(x, Y, lmbda, dim) -> float:
        """log-posterior density in the model with uniform prior on the sphere
        and asymmetric Gaussian noise. This ignores terms constant wrt x,
        since they are irrelevant for the Metropolis steps/replica swaps."""

        # Correlation is < y, x^{\otimes d} >.
        correlation = Y
        for _ in Y.shape:
            correlation = correlation @ x

        return dim * lmbda * correlation

    # Parameters
    dims = [10, 50, 100, 500]
    order = 2
    lambdas = np.logspace(np.log10(0.01), np.log10(10), 12)
    cycles = 200
    cycle_length = 100
    warmup_cycles = 20
    warmup_cycle_length = 1_000
    swap_frequency = 5  # replica swaps every .. cycles
    n_betas = 1
    betas = [round(i / n_betas, 2) for i in range(1, n_betas + 1)]
    tol = 5e-3
    tol_window = (
        100  # how long correlation has to stay inside a 2*tol interval before we stop
    )
    repetitions = 10

    pt = ParallelTempering(
        log_posterior=log_posterior,
        lmbda=lambdas[1],
        dim=dims[-1],
        order=order,
        cycles=cycles,
        warmup_cycles=warmup_cycles,
        cycle_length=cycle_length,
        warmup_cycle_length=warmup_cycle_length,
        betas=betas,
        tol=tol,
        tol_window=tol_window,
        verbose=True,
    )
    pt.run_PT()
