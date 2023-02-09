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

# from knockknock import telegram_sender
# from config import api_token, chat_id

from spiked_tensor import SpikedTensor
from unfolding import recursive_unfolding

DEFAULTS = {
    "mode": "dim",
    "d": 3,
    "n_reps": 10,
}


def run_unfolding(kwargs, seeds) -> dict:
    lmbda, dim, order = kwargs["lmbda"], kwargs["dim"], kwargs["order"]
    res = {
        "spikes": list(),
        "estimated_spikes": list(),
        "correlations": list(),
        "seeds": seeds,
    }
    for seed in seeds:
        key = PRNGKey(seed)
        key, spike, Y = SpikedTensor.generate_sample(key, lmbda, dim, order)

        estimate = recursive_unfolding(Y)
        correlation = estimate @ spike
        res["spikes"].append(spike)
        res["estimated_spikes"].append(estimate)
        res["correlations"].append(correlation)

    return res


# @telegram_sender(token=api_token, chat_id=chat_id)
def main(kwargs, mode):
    results = list()
    seeds = np.random.randint(0, 10_000, kwargs["n_reps"])
    del kwargs["n_reps"]
    if mode == "dim":
        if kwargs["order"] == 3:
            dims = [i * 50 for i in range(1, 11)]
        elif kwargs["order"] == 4:
            dims = [25, 50, 75, 100, 120]
        else:
            dims = [i * 100 for i in range(1, 11)]
        for dim in dims:
            kwargs["dim"] = dim
            res = run_unfolding(kwargs=kwargs, seeds=seeds)
            results.append(res)
        del kwargs["dim"]
        kwargs["dims"] = dims
    elif mode == "lambda":
        lambdas = np.logspace(np.log10(0.2), np.log10(10), 10)
        for lmbda in lambdas:
            kwargs["lmbda"] = lmbda
            res = run_unfolding(kwargs=kwargs, seeds=seeds)
            results.append(res)
        del kwargs["lmbda"]
        kwargs["lambdas"] = lambdas

    # Pickle parameters and results.
    kwargs["seeds"] = seeds
    data = {"params": kwargs, "results": results}
    timestring = datetime.now().strftime("%d-%m_%H:%M")
    filename = f"""data/unfolding/{mode}/{f"lambda{kwargs['lmbda']}" if mode == 'dim' else f"n{kwargs['dim']}"}_d{kwargs['order']}_{timestring}.pkl"""

    with open(filename, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices={"dim", "lambda"},
        default=DEFAULTS["mode"],
        help="Mode. Mode 'dim' gathers measurements for dim--vs-correlation plots, \
        mode 'lambda' for lambda--vs-correlation plots. Default: dim",
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
        choices=[2, 3, 4],
        default=DEFAULTS["d"],
        help=f"Tensor order. Default: {DEFAULTS['d']}.",
    )
    parser.add_argument(
        "-n_reps",
        type=int,
        default=DEFAULTS["n_reps"],
        help=f"Number of runs per (dim, lambda) pair. Default: {DEFAULTS['n_reps']}.",
    )

    kwargs = parser.parse_args().__dict__
    mode = kwargs["mode"]
    del kwargs["mode"]

    main(kwargs, mode)
