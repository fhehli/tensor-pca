from multiprocessing import Pool
from pickle import dump
from datetime import datetime

import numpy as np

from parallel_tempering import ParallelTempering


def run_paralleltempering(
    lmbda,
    dim,
    order,
    cycles,
    warmup_cycles,
    cycle_length,
    warmup_cycle_length,
    betas,
    tol,
    tol_window,
    max_lambda,
    max_dim,
    repetitions,
):
    estimated_spikes, correlations, acceptance_rates, n_swaps, runtimes = (
        [],
        [],
        [],
        [],
        [],
    )
    res = {
        "lambda": lmbda,
        "dim": dim,
        "estimated_spikes": estimated_spikes,
        "correlations": correlations,
        "acceptance_rates": acceptance_rates,
        "n_swaps": n_swaps,
        "runtimes": runtimes,
    }

    for _ in range(repetitions):
        pt = ParallelTempering(
            log_posterior=log_posterior,
            lmbda=lmbda,
            dim=dim,
            order=order,
            cycles=cycles,
            warmup_cycles=warmup_cycles,
            cycle_length=cycle_length,
            warmup_cycle_length=warmup_cycle_length,
            betas=betas,
            tol=tol,
            tol_window=tol_window,
            verbose=(lmbda == max_lambda and dim == max_dim),
        )
        pt.run_PT()
        res["estimated_spikes"].append(pt.estimate)
        res["correlations"].append(pt.correlations)
        res["acceptance_rates"].append(pt.acceptance_rate)
        res["n_swaps"].append(pt.total_swaps)
        res["runtimes"].append(pt.runtime)

    print(f"[lambda={lmbda:.2f}, dim={dim}] Done.")

    return res


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
    swap_frequency = 1  # replica swaps every .. cycles
    n_betas = 10
    betas = [round(i / n_betas, 2) for i in range(1, n_betas + 1)]
    tol = 5e-3
    tol_window = (
        20  # how long correlation has to stay inside a 2*tol interval before we stop
    )
    repetitions = 10

    max_lambda = lambdas.max()
    max_dim = max(dims)

    args = [
        [
            lmbda,
            dim,
            order,
            cycles,
            warmup_cycles,
            cycle_length,
            warmup_cycle_length,
            betas,
            tol,
            tol_window,
            max_lambda,
            max_dim,
            repetitions,
        ]
        for dim in dims
        for lmbda in lambdas
    ]

    pool = Pool()
    results = pool.starmap(
        run_paralleltempering,
        args,
    )

    time = datetime.now().strftime("%d-%m-%Y_%H:%M")

    args_filename = f"data/args/{time}.pkl"
    outfile = open(args_filename, "wb")
    dump(results, outfile)
    outfile.close()

    results_filename = f"data/corr_{time}.pkl"
    outfile = open(results_filename, "wb")
    dump(results, outfile)
    outfile.close()
