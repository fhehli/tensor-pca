import numpy as np
from numpy.linalg import norm
from numpy.random import normal
from scipy.special import gamma


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
    W = normal(0, 1/np.sqrt(n), 4*(n,))

    return lmbda*tensorize(x) + W


def sample_sphere(n) -> np.ndarray:
    """
    Get a sample drawn uniformly from the (n-1)-sphere.

    Args:
        n (int): Dimension.

    Returns:
        np.ndarray: Sample.
    """
    x = normal(0, 1, n)

    return x/norm(x)


def uniform_density_sphere(x) -> float:
    """
    Evaluate the uniform density on the sphere.

    Args:
        x (np.ndarray): Vector.

    Returns:
        float: Value of the density at x.
    """
    n = len(x)

    return gamma(n/2 + 1)/np.pi**(n/2)


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

    return -n/2*norm(x_new - x_old)**2


def get_proposal(x_old) -> np.ndarray:
    """
    Sample a vector from the proposal distribution given
    the current position of the Markov chain.

    Args:
        x_old (np.ndarray): Current vector.

    Returns:
        np.ndarray: Sample.
    """
    n = len(x_old)
    # exponent hacked to get approximately 20% acceptance rate (cf. Gelman) for n=10, lambda = 5, d=4
    x = normal(x_old, 1/n**(1.55))

    return x/norm(x)
