import argparse
import os
import pickle
import sys
from datetime import datetime

# This environment variable helps with OOM errors on GPU.
# Must be set before importing jax.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
from jax import jit
from jax.random import PRNGKey
from knockknock import telegram_sender

from config import api_token, chat_id
from parallel_tempering import ParallelTempering, SpikedTensor

DEFAULTS = {
    "m": "dim",
    "d": 3,
    "cycles": 200,
    "cycle_length": 100,
    "warmup_cycles": 15,
    "warmup_cycle_length": 1000,
    "n_betas": 10,
    "swap_frequency": 1,
    "n_reps": 10,
    "tol": 0.01,
    "tol_window": 10,
    "verbosity": 0,
}


def run_paralleltempering(kwargs, seeds) -> dict:
    lmbda, dim, order = kwargs["lmbda"], kwargs["dim"], kwargs["order"]
    res = {
        "lmbda": lmbda,
        "dim": dim,
        "spikes": list(),
        "estimated_spikes": list(),
        "correlations": list(),
        "acceptance_rates": list(),
        "runtimes": list(),
        "swap_history": list(),
        "seeds": seeds,
    }

    for seed in seeds:
        key = PRNGKey(seed)
        key, spike, Y = SpikedTensor.generate_sample(key, lmbda, dim, order)

        # Even though Y is constant, we pass it as a parameter in order to avoid
        # "constant folding", which causes unnecessary memory use in this case.
        # See https://github.com/google/jax/issues/10596#issuecomment-1119703839.
        @jit
        def log_posterior(x, Y_) -> float:
            """Log-posterior density of the model with uniform prior on the sphere
            and asymmetric Gaussian noise. This ignores terms constant wrt x,
            since they are irrelevant for the Metropolis steps/replica swaps."""

            # Correlation is < x^{\otimes d}, y >.
            correlation = Y_
            for _ in Y_.shape:
                correlation = correlation @ x

            return dim * lmbda * correlation

        pt = ParallelTempering(
            log_posterior=log_posterior,
            spike=spike,
            Y=Y,
            key=key,
            **kwargs,
        )
        pt.run_PT()
        res["spikes"].append(spike)
        res["estimated_spikes"].append(pt.estimate)
        res["correlations"].append(pt.correlations)
        res["acceptance_rates"].append(pt.acceptance_rate)
        res["runtimes"].append(pt.runtime)
        res["swap_history"].append(pt.swap_history)

    return res


@telegram_sender(token=api_token, chat_id=chat_id)
def main(kwargs, mode) -> None:
    # Run.
    results = list()
    seeds = np.random.randint(0, 10_000, kwargs["n_reps"])
    del kwargs["n_reps"]
    if mode == "dim":
        dims = (
            [i * 50 for i in range(1, 11)]
            if kwargs["order"] == 3
            else [25, 50, 75, 100, 120]
        )
        for dim in dims:
            kwargs["dim"] = dim
            res = run_paralleltempering(kwargs=kwargs, seeds=seeds)
            results.append(res)
        del kwargs["dim"]
        kwargs["dims"] = dims
    elif mode == "lambda":
        lambdas = np.logspace(np.log10(0.2), np.log10(10), 10)
        for lmbda in lambdas:
            kwargs["lmbda"] = lmbda
            res = run_paralleltempering(kwargs=kwargs, seeds=seeds)
            results.append(res)
        del kwargs["lmbda"]
        kwargs["lambdas"] = lambdas

    # Pickle parameters and results.
    kwargs["seeds"] = seeds
    data = {"params": kwargs, "results": results}
    timestring = datetime.now().strftime("%d-%m_%H:%M")
    filename = f"""data/{mode}/{f"lambda{kwargs['lmbda']}" if mode == 'dim' else f"n{kwargs['dim']}"}_d{kwargs['order']}_{timestring}.pkl"""

    with open(filename, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices={"dim", "lambda"},
        default=DEFAULTS["m"],
        help="Mode. Mode 'dim' gathers measurements for dim-running time plots, \
        mode 'lamdbda' for lambda-running time plots. Default: dim",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--dim",
        metavar="n",
        type=int,
        help="Dimension.",
        required=(sys.argv[2] == "lambda"),
    )
    parser.add_argument(
        "-l",
        "--lmbda",
        metavar="l",
        type=int,
        help="Signal-to-noise ratio.",
        required=(sys.argv[2] == "dim"),
    )
    parser.add_argument(
        "-d",
        "--order",
        metavar="d",
        type=int,
        default=DEFAULTS["d"],
        help="Tensor order. Default: 3.",
    )
    parser.add_argument(
        "-max_cycles",
        type=int,
        default=DEFAULTS["cycles"],
        help="Maximum number of samples. Default: 200.",
    )
    parser.add_argument(
        "-cycle_length",
        type=int,
        default=DEFAULTS["cycle_length"],
        help="Number of steps between samples. Default: 100.",
    )
    parser.add_argument(
        "-warmup_cycles",
        type=int,
        default=DEFAULTS["warmup_cycles"],
        help="Number of warmup cycles. Warmup steps is warmup_cycles*warmup_cycle_length. Default: 15.",
    )
    parser.add_argument(
        "-warmup_cycle_length",
        type=int,
        default=DEFAULTS["warmup_cycle_length"],
        help="Number of steps per warmup cycle. Default: 1000.",
    )
    parser.add_argument(
        "-n_betas",
        type=int,
        default=DEFAULTS["n_betas"],
        help="Number of temperatures. Default: 10.",
    )
    parser.add_argument(
        "-swap_frequency",
        type=int,
        default=DEFAULTS["swap_frequency"],
        help="How frequently to attempt replica swaps. Swaps are attempted every swap_frequency sampling cycles. Default: 5.",
    )
    parser.add_argument(
        "-n_reps",
        type=int,
        default=DEFAULTS["n_reps"],
        help="Number of runs per (dim, lambda) pair. Default: 10.",
    )
    parser.add_argument(
        "-tol",
        type=float,
        default=DEFAULTS["tol"],
        help="Tolerance window used to check for convergence. Default: 0.01.",
    )
    parser.add_argument(
        "-tol_window",
        type=int,
        default=DEFAULTS["tol_window"],
        help="How many cycles correlation has to stay inside a 2*tol interval before stopping. Default: 10.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=DEFAULTS["verbosity"],
        help="Verbosity. Set to a truthy to activate verbose mode. Default: 0.",
    )

    kwargs = parser.parse_args().__dict__
    mode = kwargs["mode"]
    del kwargs["mode"]

    main(kwargs, mode)
