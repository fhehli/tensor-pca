from time import perf_counter

import jax.numpy as np
from jax.random import choice, exponential, normal, split, PRNGKey
from tqdm import tqdm

from utils import d_fold_tensor_product, get_normal_proposal, sample_sphere


class SpikedTensor:
    def __init__(
        self,
        lmbda,
        dim,
        order=4,
        seed=0,
    ) -> None:

        # Signal-to-noise parameter
        self.lmbda = lmbda

        # Dimension
        self.dim = dim

        # Tensor order
        self.order = order

        # Fix random seed
        key = PRNGKey(seed)

        # Generate a sample
        self.spike, self.Y = SpikedTensor.generate_sample(
            key, self.lmbda, self.dim, self.order
        )

    @staticmethod
    def generate_sample(
        key, lmbda, n, d
    ) -> list[np.DeviceArray, np.DeviceArray, np.DeviceArray]:
        """Generates a sample of the form Y = lambda*x^{\otimes d} + W/sqrt(n)"""
        key, subkey = split(key)
        spike = sample_sphere(subkey, n)  # sampled uniformly from the sphere
        key, subkey = split(key)
        W = normal(subkey, d * (n,))  # iid standard normal noise
        Y = lmbda * d_fold_tensor_product(spike, d) + W / np.sqrt(n)

        return key, spike, Y


class ParallelTempering:
    def __init__(
        self,
        log_posterior,
        spike,
        Y,
        dim,
        lmbda,
        key,
        order=4,
        cycles=1_000,
        cycle_length=100,
        warmup_cycles=50,
        warmup_cycle_length=1000,
        n_betas=10,
        swap_frequency=1,
        get_proposal=get_normal_proposal,
        tol=5e-3,
        tol_window=10,
        verbose=False,
        store_chain=False,
    ) -> None:
        self.log_posterior = log_posterior
        self.spike = spike
        self.Y = Y
        self.dim = dim
        self.lmbda = lmbda
        self.key = key

        # Tensor order.
        self.order = order

        # Proposal sampler
        self.get_proposal = get_proposal

        # Number of cycles for MCMC iteration
        self.cycles = cycles
        self.cycle_length = cycle_length

        # Number of warmup cycles
        self.warmup_cycles = warmup_cycles
        self.warmup_cycle_length = warmup_cycle_length

        # Inverse temperatures
        self.n_betas = n_betas
        self.betas = [round(i / n_betas, 2) for i in range(1, n_betas + 1)]

        # How frequently to attempt replica swaps
        self.swap_frequency = swap_frequency

        # Stopping tolerance
        self.tol = tol
        self.tol_window = tol_window

        self.verbose = verbose
        if verbose:
            self.verb_prefix = f"[lambda={lmbda:.1f}, dim={self.dim}]"
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
        self.key, *subkeys = split(self.key, n_betas + 1)
        self.current_state = {
            beta: sample_sphere(subkey, self.dim)
            for subkey, beta in zip(subkeys, self.betas)
        }

        if store_chain:
            # Prepare storage for sample chain
            self.chain = np.zeros((self.cycles, self.dim))
            self.chain[0] = self.current_state[1]

        # Inner products of the spike and the estimated spikes, updated after each cycle
        self.correlations = np.zeros(self.cycles)

        # We store for each temperature and each replica swap whether it swapped
        # with the above (+1) or below chain (-1). If no swap happened for temper-
        # ature beta in replica swap i, then we set it to zero.
        self.swap_history = dict.fromkeys(
            self.betas, np.zeros(self.cycles // swap_frequency)
        )

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

    def replica_swaps(self, key, i) -> None:
        # Decide to swap replica i and i+1 for even or odd i:
        # parity=0 corresponds to even, parity=1 to odd
        key, subkey = split(key)
        parity = choice(subkey, 2)

        for smaller_beta, bigger_beta in zip(
            self.betas[parity::2], self.betas[(parity + 1) :: 2]
        ):
            # (log) acceptance probability
            r = (smaller_beta - bigger_beta) * (
                self.log_posterior(self.current_state[bigger_beta], self.Y)
                - self.log_posterior(self.current_state[smaller_beta], self.Y)
            )

            # Equivalent to np.random.uniform() < exp(r).
            if -exponential(key) < r:
                # Swap states.
                self.current_state[smaller_beta], self.current_state[bigger_beta] = (
                    self.current_state[bigger_beta],
                    self.current_state[smaller_beta],
                )
                # Record swap.
                self.swap_history[smaller_beta] = (
                    self.swap_history[smaller_beta].at[i].set(1)
                )
                self.swap_history[bigger_beta] = (
                    self.swap_history[bigger_beta].at[i].set(-1)
                )

    def mh_step(self, key, x_old, beta) -> list[np.DeviceArray, int]:
        """Takes one Metropolis step. Assumes proposal density is symmetric."""
        key, subkey = split(key)
        proposal = self.get_proposal(subkey, x_old, self.scaling_parameters[beta])
        r = beta * (
            self.log_posterior(proposal, self.Y) - self.log_posterior(x_old, self.Y)
        )

        # Equivalent to np.random.uniform() < exp(r).
        if -exponential(key) < r:
            return proposal, 1
        else:
            return x_old, 0

    def run_cycle(
        self, key, cycle_length=1_000, beta=1
    ) -> list[np.DeviceArray, np.DeviceArray, int]:
        """Takes cycle_length many Metropolis steps.

        Returns new state and acceptance rate."""
        n_accepted = 0
        x = self.current_state[beta]
        for _ in range(cycle_length):
            key, subkey = split(key)
            x, accepted = self.mh_step(subkey, x, beta)
            n_accepted += accepted

        return x, n_accepted / cycle_length

    def warmup(self, beta) -> float:
        """Runs warmup cycles for one temperature."""
        iterator = (
            tqdm(
                range(self.warmup_cycles),
                desc=f"{self.verb_prefix} beta={beta:.2f} WARMUP",
            )
            if self.verbose
            else range(self.warmup_cycles)
        )

        for _ in iterator:
            self.key, subkey = split(self.key)
            self.current_state[beta], acceptance_rate = self.run_cycle(
                subkey,
                self.warmup_cycle_length,
                beta,
            )

            # Update scaling of proposal distribution to improve acceptance rate.
            factor = self.get_update_factor(acceptance_rate)
            self.scaling_parameters[beta] *= factor

        if self.verbose:
            print(
                f"{self.verb_prefix} Finished warmup cycles for beta={beta:.2f}. Final acceptance rate was {int(100*acceptance_rate)}%."
            )

        return self.scaling_parameters[beta]

    def run_PT(self) -> None:
        start_time = perf_counter()
        if self.verbose:
            print(f"{self.verb_prefix} Starting warmup cycles.")

        # Warmup cycles for all temperatures.
        for beta in self.betas:
            self.warmup(beta)

        ## SAMPLING ##
        if self.verbose:
            print(f"{self.verb_prefix} Finished warmup. Starting sampling.")
        self.acceptance_rate = 0

        # Define an iterator with progress bar if in verbose mode.
        cycle_iterator = (
            tqdm(
                range(1, self.cycles + 1),
                desc=f"{self.verb_prefix} SAMPLING",
            )
            if self.verbose
            else range(1, self.cycles + 1)
        )
        # Run cycles.
        for i in cycle_iterator:
            for beta in self.betas:
                self.key, subkey = split(self.key)
                self.current_state[beta], acceptance_rate = self.run_cycle(
                    subkey, self.cycle_length, beta
                )

            # Update acceptance rate for beta=1.
            self.acceptance_rate *= i - 1
            self.acceptance_rate += acceptance_rate
            self.acceptance_rate /= i

            # Update states and perform replica swaps.
            if i % self.swap_frequency == 0:
                self.key, subkey = split(self.key)
                self.replica_swaps(subkey, i)

            # Update estimated spike, correlations and save sample.
            self.estimate *= i
            self.estimate += self.current_state[1]
            self.estimate /= i + 1
            correlation = self.estimate @ self.spike
            self.correlations = self.correlations.at[i - 1].set(correlation)
            if self.store_chain:
                self.chain = self.chain.at[i - 1].set(self.current_state[1])

            # Check "convergence":
            # We check whether the correlation of the last n samples lies
            # within an interval of size 2*self.tol, where n = self.tol_window.
            if i >= self.tol_window and np.alltrue(
                np.abs(
                    self.correlations[i - self.tol_window : i]
                    - self.correlations[i - self.tol_window : i].mean()
                )
                < self.tol
            ):
                if self.verbose:
                    print(
                        f"{self.verb_prefix} Correlation has converged after {i} cycles."
                    )
                # Omit non-measured correlations.
                self.correlations = self.correlations[:i]
                self.swap_history = {
                    beta: self.swap_history[beta][:i] for beta in self.betas
                }
                break  # Stop sampling loop.

            # Print message every other cycle.
            if self.verbose and (i % (self.cycles // 10) == 0):
                print(
                    f"{self.verb_prefix} Finished {i} cycles. Current correlation is {correlation:.2f}. Acceptance rate so far is {int(100*self.acceptance_rate)}%."
                )

        end_time = perf_counter()
        self.runtime = end_time - start_time
        if self.verbose:
            print(
                f"{self.verb_prefix} Finished sampling. Correlation was {self.correlations[-1]:.2f}. Final acceptance rate was {int(100*self.acceptance_rate)}%. There were {np.abs(self.swap_history[1]).sum()} swaps. Runtime was {self.runtime:.0f}s."
            )
