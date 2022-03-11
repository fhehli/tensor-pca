from warnings import simplefilter

import numpy as np
from numpy.linalg import norm
from tqdm import tqdm

from utils import sample_sphere, sample_spiked_tensor, tensorize, get_proposal, log_proposal_density


def metropolis_hastings(log_scaled_posterior,
                        starting_point,
                        N=10000
                        ) -> list[list, int]:
    """
    Metropolis-Hastings algorithm (Sec. 11.2; Bayesian Data Analysis; Gelman, A.)

    Args:
        log_scaled_posterior : Natural log of the numerator of the posterior density.
        starting_point (np.ndarray): Starting point of the Markov chain.
        N (int, optional): _description_. Defaults to 10000.

    Returns:
        list[list, int]: Sample chain, Number of accepted proposals
    """
    chain = [starting_point]
    accepted = 0

    for i in tqdm(range(N)):
        x_old = chain[-1]
        x_new = get_proposal(x_old)

        r = log_scaled_posterior(x_new) - \
            log_scaled_posterior(
                x_old)  # proposal density can be cancelled since it's symmetric in x_new, x_old

        if np.log(np.random.uniform()) < r:
            chain.append(x_new)
            accepted += 1
        else:
            chain.append(chain[-1])

    return chain, accepted


if __name__ == "__main__":
    # simplefilter("ignore")

    # parameters
    d = 4           # tensor order (don't change)
    lmbda = 5       # signal-to-noise ratio
    n = 10          # dimension

    x = sample_sphere(n)
    Y = sample_spiked_tensor(lmbda, x)  # Y = lambda*x^{\otimes d} + W

    def log_scaled_posterior(x) -> float:
        # Y|x ~ N(lambda*x, 1/n),  x ~ U(S^{n-1})

        return -n/2*norm(Y - lmbda*tensorize(x))**2

    N = 10000       # number of steps
    chain, accepted = metropolis_hastings(
        log_scaled_posterior, sample_sphere(n), N)

    x_hat = np.mean(chain, axis=0)  # empirical posterior mean

    print("\n||x_hat - x||: ", norm(x_hat - x))
    print("< x_hat, x > : ", x_hat @ x)
    print(
        f"Acceptance rate: {int(100*accepted/N)}%")
