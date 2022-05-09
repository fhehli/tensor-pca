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
            lmbda=lmbda,
            dim=dim,
            order=order,
            cycles=cycles,
            warmup_cycles=warmup_cycles,
            cycle_length=cycle_length,
            warmup_cycle_length=warmup_cycle_length,
            betas=betas,
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
    # parameters
    dims = [10, 25, 50, 75]
    order = 3
    lambdas = np.logspace(np.log10(0.5), np.log10(10), 10)
    cycles = 200
    cycle_length = 100
    warmup_cycles = 20
    warmup_cycle_length = 1_000
    betas = [0.1 * i for i in range(1, 11)]
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

    filename = f"data/corr_{datetime.now().strftime('_%d-%m-%Y_%H:%M')}.pkl"
    outfile = open(filename, "wb")
    dump(results, outfile)
    outfile.close()
