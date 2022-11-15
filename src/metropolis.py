from time import perf_counter

import jax.numpy as np
from jax.numpy.linalg import norm
from jax import random
from tqdm import tqdm

from utils import get_normal_proposal, sample_sphere

class Metropolis:
    """
    Implementation of Metropolis MCMC for the spiked tensor problem.
    """

    def __init__(
        self,
        log_posterior,
        spike,
        Y,
        dim,
        lmbda,
        key,
        order=4,
        max_cycles=1_000,
        cycle_length=100,
        warmup_cycles=50,
        warmup_cycle_length=1000,
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

        # Proposal sampler.
        self.get_proposal = get_proposal

        # Number of cycles for MCMC iteration.
        self.max_cycles = max_cycles
        self.cycle_length = cycle_length

        # Number of warmup cycles.
        self.warmup_cycles = warmup_cycles
        self.warmup_cycle_length = warmup_cycle_length

        # Stopping tolerance.
        self.tol = tol
        self.tol_window = tol_window

        self.verbose = verbose
        if verbose:
            self.verb_prefix = f"[lambda={lmbda:.1f}, dim={self.dim}]"
        self.store_chain = store_chain

        # The acceptance rate over all sampling steps.
        self.acceptance_rate = 0

        self.acceptance_rate_history = list()

        # How long self.run() took.
        self.runtime = None

        # Storage for scaling parameters (variance) of jumping distribution.
        self.scaling_parameters = 1.0

        # Estimated spike (mean).
        self.estimate = np.zeros(self.dim)

        # Initialize starting state.
        self.key, subkey = random.split(self.key)
        self.current_state = sample_sphere(subkey, dim)

        if store_chain:
            # Prepare storage for sample chain.
            self.chain = np.zeros((self.max_cycles, dim))
            self.chain[0] = self.current_state[1]

        # Inner products of the spike and the estimated spikes, updated after each cycle.
        self.correlations = np.zeros(self.max_cycles)

    def _get_update_factor(acceptance_rate) -> float:
        """Returns a factor to update the scaling parameters of
        the proposal distribution. We aim for an acceptance rate
        between 20% and 30%."""
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

    def _mh_step(self, key, x_old) -> tuple[np.DeviceArray, int]:
        """Takes one Metropolis step. Assumes proposal density is symmetric."""
        key, subkey = random.split(key)
        proposal = self.get_proposal(subkey, x_old, self.scaling_parameters)
        r = self.log_posterior(proposal, self.Y) - self.log_posterior(x_old, self.Y)

        # Equivalent to np.random.uniform() < exp(r).
        if -random.exponential(key) < r:
            return proposal, 1
        else:
            return x_old, 0

    def _run_cycle(self, key, cycle_length) -> tuple[np.DeviceArray, float]:
        """Takes cycle_length many Metropolis steps."""
        n_accepted = 0
        x = self.current_state
        for _ in range(cycle_length):
            key, subkey = random.split(key)
            x, accepted = self._mh_step(subkey, x)
            n_accepted += accepted

        return x, n_accepted / cycle_length

    def _warmup(self) -> float:
        """Runs warmup cycles for one temperature."""
        n_cycles = (
            tqdm(
                range(self.warmup_cycles),
                desc=f"{self.verb_prefix} WARMUP",
            )
            if self.verbose
            else range(self.warmup_cycles)
        )

        for _ in n_cycles:
            self.key, subkey = random.split(self.key)
            self.current_state, acceptance_rate = self._run_cycle(
                subkey,
                self.warmup_cycle_length,
            )

            # Update scaling of proposal distribution to improve acceptance rate.
            factor = self._get_update_factor(acceptance_rate)
            self.scaling_parameters *= factor

        if self.verbose:
            print(
                f"{self.verb_prefix} Finished warmup cycles. Final acceptance rate was {int(100*acceptance_rate)}%."
            )

    def run(self) -> None:
        if self.verbose:
            print(f"{self.verb_prefix} Starting warmup cycles.")

        start_time = perf_counter()
        # Warmup.
        self._warmup()

        ## SAMPLING ##
        if self.verbose:
            print(f"{self.verb_prefix} Finished warmup. Starting sampling.")

        # Define an iterator with progress bar if in verbose mode.
        n_cycles = (
            tqdm(
                range(1, self.max_cycles + 1),
                desc=f"{self.verb_prefix} SAMPLING",
            )
            if self.verbose
            else range(1, self.max_cycles + 1)
        )
        for i in n_cycles:
            # Run cycles.
            self.key, subkey = random.split(self.key)
            self.current_state, acceptance_rate = self._run_cycle(
                subkey, self.cycle_length
            )

            # Update acceptance rate.
            self.acceptance_rate_history.append(acceptance_rate)
            self.acceptance_rate = np.mean(np.array(self.acceptance_rate_history))

            # Update estimated spike, correlations and save sample.
            """
            Antoine: Write that we take the mean as estimator, we could also e.g. take the last sample, here the mean is more coherent choice 
            since we want to compute the posterior average.
            """
            self.estimate *= i
            self.estimate += self.current_state
            self.estimate /= i + 1
            self.estimate /= norm(self.estimate)

            correlation = self.current_state @ self.spike
            self.correlations = self.correlations.at[i - 1].set(correlation)
            if self.store_chain:
                self.chain = self.chain.at[i - 1].set(self.current_state)

            # Check "convergence":
            # We check whether the correlation of the last n samples lies
            # within an interval of size 2*self.tol, where n = self.tol_window
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
                break  # Stop sampling loop.

            # Print message every other cycle.
            if self.verbose and (i % (self.max_cycles // 10) == 0):
                print(
                    f"{self.verb_prefix} Finished {i} cycles. Current correlation is {correlation:.2f}. Acceptance rate so far is {int(100*self.acceptance_rate)}%."
                )

        self.runtime = perf_counter() - start_time
        if self.verbose:
            print(
                f"{self.verb_prefix} Finished sampling. Correlation was {self.correlations[-1]:.2f}. Final acceptance rate was {int(100*self.acceptance_rate)}%. There were {np.abs(self.swap_history[1]).sum()} swaps. Runtime was {self.runtime:.0f}s."
            )
