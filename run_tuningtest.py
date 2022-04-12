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
    repetitions,
):
    st = ParallelTempering(
        lmbda=lmbda,
        dim=dim,
        order=order,
        cycles=cycles,
        warmup_cycles=warmup_cycles,
        cycle_length=cycle_length,
        warmup_cycle_length=warmup_cycle_length,
        betas=betas,
    )
    print(f"Starting run of lambda={lmbda} in dim={dim}.")

    res = dict.fromkeys(range(repetitions))

    for i in range(repetitions):
        st.run_PT()
        res[i] = {
            "lambda": lmbda,
            "dim": dim,
            "estimated spike": st.estimate,
            "correlation": st.correlation,
            "runtime": st.runtime,
            "acceptance_rate": st.acceptance_rate,
        }
    print(f"Finished run of lambda={lmbda} in dim={dim}.")

    return res


if __name__ == "__main__":
    # # parameters
    dims = [8, 10, 12]
    order = 4
    asymptotic_lambdas = {n: n ** ((order - 2) / 4) for n in dims}
    lambdas = {
        n: np.linspace(asymptotic_lambdas[n] - 1, asymptotic_lambdas[n] + 5, 8)
        for n in dims
    }
    cycles = 1_000
    cycle_length = 10
    warmup_cycles = 50
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
                repetitions,
            ]
            for dim in dims
            for lmbda in lambdas[dim]
        ],
    )

    filename = "data/pt_results" + datetime.now().strftime("_%d-%m-%Y_%H:%M") + ".pkl"
    outfile = open(filename, "wb")
    dump(results, outfile)
    outfile.close()
