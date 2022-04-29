from multiprocessing import Pool
from pickle import dump
from datetime import datetime

import numpy as np

from parallel_tempering import ParallelTempering


def run_paralleltempering(
    lmbda, dim, order, cycles, warmup_cycles, cycle_length, warmup_cycle_length, betas
):
    pt = ParallelTempering(
        lmbda=lmbda,
        dim=dim,
        order=order,
        cycles=cycles,
        warmup_cycles=warmup_cycles,
        cycle_length=cycle_length,
        warmup_cycle_length=warmup_cycle_length,
        betas=betas,
    )
    estimated_spikes, correlations, acceptance_rates = [], [], []
    res = [lmbda, dim, estimated_spikes, correlations, acceptance_rates]
    print(f"Starting run of lambda={lmbda} in dim={dim}.")
    for i in range(repetitions):
        pt.estimate = np.zeros(dim)
        pt.acceptance_rate = 0
        pt.run_PT()
        estimated_spikes.append(pt.estimate)
        correlations.append(pt.correlation)
        acceptance_rates.append(pt.acceptance_rate)

    print(f"Finished run of lambda={lmbda} in dim={dim}.")
    return res


if __name__ == "__main__":
    # parameters
    dims = [25, 50]
    order = 3
    lambdas = np.logspace(np.log10(2), np.log10(10), 10)
    cycles = 500
    warmup_cycles = 500
    cycle_length = 100
    warmup_cycle_length = 100
    betas = [0.1 * i for i in range(1, 11)]
    repetitions = 5

    pool = Pool()
    results = pool.starmap(
        run_paralleltempering,
        [
            [
                lmbda,
                dim,
                order,
                cycles,
                warmup_cycles,
                cycle_length,
                warmup_cycle_length,
                betas,
            ]
            for dim in dims
            for lmbda in lambdas[dim]
        ],
    )

    filename = f"data/corr_{datetime.now().strftime('_%d-%m-%Y_%H:%M')}.pkl"
    outfile = open(filename, "wb")
    dump(results, outfile)
    outfile.close()
