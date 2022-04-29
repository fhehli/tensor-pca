import numpy as np
from numpy.linalg import norm
from numpy.random import normal


def tensorize(x, d=4) -> np.ndarray:
    """
    Compute d-fold tensor product of a vector.

    Args:
        x (np.ndarray): Vector
        d (int): Order of tensor product
    Returns:
        np.ndarray: d-fold tensor product of x.
    """
    assert d > 1, "Error (in call to tensorize): Tensor order must be bigger than 1."

    y = np.tensordot(x, x, axes=0)
    while d > 2:
        y = np.tensordot(y, x, axes=0)
        d -= 1

    return y


def normalise(x) -> np.ndarray:
    return x / norm(x)


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

    return -n / 2 * sum((x_new - x_old) ** 2)


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
