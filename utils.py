import numpy as np
from numpy.linalg import norm
from numpy.random import normal


def tensorize(x) -> np.ndarray:
    """
    Compute 4-fold tensor product of a vector.

    Args:
        x (np.ndarray): Vector

    Returns:
        np.ndarray: 4-fold tensor product of x.
    """
    return np.einsum("i,j,k,l->ijkl", x, x, x, x)


def sample_spiked_tensor(lmbda, x) -> np.ndarray:
    """
    Get a sample of the spiked tensor model.

    Args:
        lmbda (float): Signal-to-noise parameter.
        x (vector): The spike.

    Returns:
        np.ndarray: Sample.
    """
    n = len(x)
    W = normal(0, 1 / np.sqrt(n), 4 * (n,))

    return lmbda * tensorize(x) + W


def sample_sphere(n) -> np.ndarray:
    """
    Get a sample drawn uniformly from the (n-1)-sphere.

    Args:
        n (int): Dimension.

    Returns:
        np.ndarray: Sample.
    """
    x = normal(0, 1, n)

    return x / norm(x)


def log_proposal_density(x_new, x_old) -> float:
    """
    Evaluate the natural log of the (numerator of the)
    density of the proposal distribution.

    Args:
        x_new (np.ndarray): Proposal vector.
        x_old (np.ndarray): Current vector.

    Returns:
        float: Value at x_new given x_old.
    """
    n = len(x_old)

    return -n / 2 * norm(x_new - x_old) ** 2


def get_normal_proposal(x_old, scaling_parameter=1) -> np.ndarray:
    """
    Sample a vector from a normal distribution centered at the
    current position of the Markov chain.

    Args:
        x_old (np.ndarray): Current vector.

    Returns:
        np.ndarray: Sample.
    """
    n = len(x_old)
    x = normal(x_old, (1 / n) ** scaling_parameter)

    return x / norm(x)


def mh_step(x_old, log_posterior, get_proposal) -> list[np.ndarray, int]:
    """
    Compute next state for Metropolis-Hastings.

    Args:
        x_old (np.ndarray): Current state.
        log_posterior (callable) : Natural log of the posterior density.
        get_proposal (callable) : Provides a proposal state.

    Returns:
        np.ndarray, int: New state, Indicator of acceptance..
    """
    proposal = get_proposal(x_old)

    r = log_posterior(proposal) - log_posterior(x_old)
    # proposal density can be cancelled since it's symmetric in x_new, x_old

    if np.log(np.random.uniform()) < r:
        return proposal, 1
    else:
        return x_old, 0


def update_scaling_parameter(beta, acceptance_ratio, scaling_parameter) -> float:
    """
    Returns an updated scaling parameter. The update is
    computed based on the acceptance ratio in the last
    steps. The goal is an acceptance ratio of ~23%.

    Args:
        beta (float): Temperature parameter.
        acceptance_ratio (float): Acceptance ratio of
                                  the last few steps.
        scaling_parameter (float): Old scaling parameter.

    Returns:
        scaling_parameter: The updated scaling parameter.
    """

    if acceptance_ratio < 0.20:
        scaling_parameter *= 1.1
    elif acceptance_ratio > 0.25:
        scaling_parameter *= 0.9

    return scaling_parameter
