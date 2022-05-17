import argparse
import os
import pickle
from datetime import datetime

# Must be set before importing jax.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
from jax import jit
from jax.random import PRNGKey
from knockknock import telegram_sender

from config import api_token
from parallel_tempering import ParallelTempering, SpikedTensor


def run_paralleltempering(lmbda, kwargs, seeds):
    res = {
        "lmbda": lmbda,
        "dim": kwargs["dim"],
        "spikes": [],
        "estimated_spikes": [],
        "correlations": [],
        "acceptance_rates": [],
        "runtimes": [],
        "swap_history": [],
        "seeds": seeds,
    }
    dim = kwargs["dim"]
    order = kwargs["order"]

    for seed in seeds:
        key = PRNGKey(seed)
        key, spike, Y = SpikedTensor.generate_sample(key, lmbda, dim, order)

        # Even though Y is constant, we pass it as a parameter in order to avoid
        # jit "baking it into" the compiled function (constant folding), and
        # as a result causing unnecessary memory use.
        # See https://github.com/google/jax/issues/10596#issuecomment-1119703839.
        @jit
        def log_posterior(x, y) -> float:
            """Log-posterior density of the model with uniform prior on the sphere
            and asymmetric Gaussian noise. This ignores terms constant wrt x,
            since they are irrelevant for the Metropolis steps/replica swaps."""

            # Correlation is < y, x^{\otimes d} >.
            correlation = y
            for _ in y.shape:
                correlation = correlation @ x

            return dim * lmbda * correlation

        pt = ParallelTempering(
            log_posterior=log_posterior,
            spike=spike,
            Y=Y,
            lmbda=lmbda,
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


# To get notifications when main() returns, change the token
# and chat_id in the below decorator by following the steps at
# https://github.com/huggingface/knockknock#telegram.
@telegram_sender(token=api_token, chat_id=1533966132)
def main(kwargs):
    n_lambdas = kwargs["n_lambdas"]
    n_runs = kwargs["n_runs"]

    lambdas = np.logspace(np.log10(0.2), np.log10(10), n_lambdas)
    seeds = list(range(n_runs))

    # We remove these so that we can pass kwargs to run_paralleltempering.
    del kwargs["n_lambdas"], kwargs["n_runs"]

    # Run.
    results = []
    for lmbda in lambdas:
        res = run_paralleltempering(lmbda=lmbda, kwargs=kwargs, seeds=seeds)
        results.append(res)

    timestring = datetime.now().strftime("%d-%m-%Y_%H:%M")

    # Save arguments.
    kwargs["lambdas"] = lambdas
    kwargs["seeds"] = seeds
    args_filename = f"data/params/{timestring}.pkl"
    with open(args_filename, "wb") as f:
        pickle.dump(kwargs, f)

    # Save results.
    dim = kwargs["dim"]
    order = kwargs["order"]
    results_filename = f"data/n{dim}_d{order}_{timestring}.pkl"
    with open(results_filename, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--dim",
        metavar="n",
        type=int,
        nargs="?",
        default=10,
        help="Dimension. Default: 10.",
    )
    parser.add_argument(
        "-d",
        "--order",
        metavar="d",
        type=int,
        nargs="?",
        default=4,
        help="Tensor order. Default: 4.",
    )
    parser.add_argument(
        "-n_lambdas",
        type=int,
        # nargs="?",
        default=10,
        help="Number of lambdas. Default: 10.",
    )
    parser.add_argument(
        "-cycles",
        type=int,
        # nargs="?",
        default=200,
        help="Number of samples. Default: 200.",
    )
    parser.add_argument(
        "-cycle_length",
        type=int,
        # nargs="?",
        default=100,
        help="Number of steps between samples. Default: 100.",
    )
    parser.add_argument(
        "-warmup_cycles",
        type=int,
        # nargs="?",
        default=10,
        help="Number of warmup cycles. Warmup steps is warmup_cycles*warmup_cycle_length. Default: 10.",
    )
    parser.add_argument(
        "-warmup_cycle_length",
        type=int,
        # nargs="?",
        default=1000,
        help="Number of steps per warmup cycle. Default: 1000.",
    )
    parser.add_argument(
        "-n_betas",
        type=int,
        # nargs="?",
        default=10,
        help="Number of temperatures. Default: 10.",
    )
    parser.add_argument(
        "-swap_frequency",
        type=int,
        # nargs="?",
        default=5,
        help="How frequently to attempt replica swaps. Swaps are attempted every swap_frequency sampling cycles. Default: 5.",
    )
    parser.add_argument(
        "-n_runs",
        type=int,
        # nargs="?",
        default=5,
        help="Number of runs per (dim, lambda) pair. Default: 5.",
    )
    parser.add_argument(
        "-tol",
        type=float,
        # nargs="?",
        default=5e-3,
        help="Tolerance window. Default: 0.01.",
    )
    parser.add_argument(
        "-tol_window",
        type=int,
        # nargs="?",
        default=5,
        help="How many cycles correlation has to stay inside a 2*tol interval before stopping. Default: 5.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        # nargs="?",
        default=0,
        help="Prints some information when set to a thruthy. Default: 0.",
    )

    kwargs = parser.parse_args().__dict__
    main(kwargs)
