from functools import partial

import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

from utils import (
    sample_sphere,
    sample_spiked_tensor,
    tensorize,
    get_normal_proposal,
    mh_step,
    update_scaling_parameter,
)


def parallel_tempering(
    log_posterior,
    betas,
    steps=10000,
    get_proposal=get_normal_proposal,
    burn_in_steps=1000,
) -> list[list, int]:
    """
    Parallel Tempering algorithm
    (cf. https://pubs.rsc.org/en/content/articlelanding/2005/cp/b509983h)

    Args:
        log_posterior (callable): Natural log of the posterior density.
        betas (list): Defines the 'temperatures'.
        steps (int, optional): Number of steps. Defaults to 10000.
        get_proposal (callable, optional): Provides proposal states. Defaults
                                           to get_normal_proposal.

    Returns:
        list[list, int]: Sample chain for beta=1, Number of accepted proposals.
    """

    # initialize chains and set starting points randomly
    chains = {beta: [sample_sphere(n)] for beta in betas}

    # define densities
    log_posteriors = {beta: lambda x: log_posterior(x) ** beta for beta in betas}

    # storage for acceptance counters
    acceptance_counters = dict.fromkeys(betas, 0)

    # storage for scaling parameters of proposal getters
    scaling_parameters = dict.fromkeys(betas, 1.0)

    # take steps
    for i in tqdm(range(steps)):
        for beta in betas:
            x_old = chains[beta][-1]

            # get new state
            x_new, accepted = mh_step(
                x_old,
                log_posteriors[beta],
                partial(get_proposal, scaling_parameter=scaling_parameters[beta]),
            )

            # update chain
            chains[beta].append(x_new)
            acceptance_counters[beta] += accepted

            # update proposal sampler scaling every 100 steps
            # to optimize acceptance rate
            if 0 < i <= burn_in_steps and i % 100 == 0:
                scaling_parameters[beta] = update_scaling_parameter(
                    beta, acceptance_counters[beta] / 100, scaling_parameters[beta]
                )
                acceptance_counters[beta] = 0  # reset counter

        # perform replica swaps every fifth step
        if i % 5 == 0:
            # randomly choose to swap replicas i and i+1 for even or odd i
            parity = np.random.choice([0, 1])
            # iterate over (i,i+1) pairs
            for smaller_beta, bigger_beta in zip(
                betas[parity::2], betas[(parity + 1) :: 2]
            ):
                # compute acceptance probability
                log_acceptance_probability = (bigger_beta - smaller_beta) * (
                    log_posterior(chains[bigger_beta][-1])
                    - log_posterior(chains[smaller_beta][-1])
                )
                if np.log(np.random.uniform()) < log_acceptance_probability:
                    # swap states i and i+1
                    chains[smaller_beta][-1], chains[bigger_beta][-1] = (
                        chains[bigger_beta][-1],
                        chains[smaller_beta][-1],
                    )

    # return chain for beta=1
    return chains[1], acceptance_counters[1]


if __name__ == "__main__":

    # parameters
    d = 4  # tensor order (don't change)
    lmbda = 5  # signal-to-noise ratio
    n = 10  # dimension
    betas = [0.05 * i for i in range(1, 21)]

    x = sample_sphere(n)
    Y = sample_spiked_tensor(lmbda, x)  # Y = lambda*x^{\otimes d} + W

    def log_posterior(x) -> float:
        # Y|x ~ N(lambda*x, 1/n),  x ~ U(S^{n-1})

        return -n / 2 * norm(Y - lmbda * tensorize(x)) ** 2

    steps = 10000  # number of steps
    chain, accepted = parallel_tempering(log_posterior, betas, steps=steps)

    x_hat = np.mean(chain[(steps // 10) :: 2], axis=0)  # estimated posterior mean

    print("\n||x_hat - x||: ", norm(x_hat - x))
    print("< x_hat, x > : ", x_hat @ x)
    print(f"Acceptance rate: {int(100 * accepted / steps)}%")
