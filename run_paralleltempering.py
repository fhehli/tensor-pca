from multiprocessing import Pool
from pickle import dump
from datetime import datetime

import numpy as np

from parallel_tempering import SpikedTensor


def run_paralleltempering(
    lmbda, dim, order, cycles, warmup_cycles, cycle_length, betas
):
    st = SpikedTensor(
        lmbda=lmbda,
        dim=dim,
        order=order,
        cycles=cycles,
        warmup_cycles=warmup_cycles,
        cycle_length=cycle_length,
        betas=betas,
    )
    st.run_PT()
    estimated_spike = st.estimate
    correlation = st.correlation
    runtime = st.runtime

    return {
        "lambda": lmbda,
        "dim": dim,
        "estimated spike": estimated_spike,
        "correlation": correlation,
        "runtime": runtime,
    }


if __name__ == "__main__":
    # # parameters
    dims = [10, 20]
    order = 4
    asymptotic_lambdas = {n: n ** ((order - 2) / 4) for n in dims}
    lambdas = {
        n: np.linspace(asymptotic_lambdas[n] - 1, asymptotic_lambdas[n] + 5, 50)
        for n in dims
    }
    cycles = 1_000
    warmup_cycles = 100
    cycle_length = 100
    betas = [0.05 * i for i in range(1, 21)]

    pool = Pool()
    results = pool.starmap(
        run_paralleltempering,
        [
            [lmbda, dim, order, cycles, warmup_cycles, cycle_length, betas]
            for dim in dims
            for lmbda in lambdas[dim]
        ],
    )

    filename = "data/pt_results" + datetime.now().strftime("_%d-%m-%Y_%H:%M") + ".pkl"
    outfile = open(filename, "wb")
    dump(results, outfile)
    outfile.close()
