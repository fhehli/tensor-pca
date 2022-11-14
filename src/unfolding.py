import jax.numpy as np
from jax.numpy.linalg import svd


def recursive_unfolding(Y: np.DeviceArray) -> np.DeviceArray:
    """
    Estimate the spike according to recursive tensor unfolding.
    (Sec 3.2 in https://arxiv.org/abs/1411.1076.)

    Args:
        Y (np.DeviceArray): Spiked tensor sample.

    Returns:
        np.DeviceArray: Estimated spike.
    """
    n = Y.shape[0]  # Dimension.
    d = len(Y.shape)  # Tensor order.
    Mat_Y = Y.reshape(n ** np.floor(d / 2), n ** np.ceil(d / 2))
    *_, Vh = svd(Mat_Y, full_matrices=False)
    w = Vh[0]  # w is the top right-singular vector of Mat_Y.
    Mat_w = w.reshape(n, n ** (np.ceil(d / 2) - 1))
    U, *_ = svd(Mat_w, full_matrices=False)
    estimate = U[:, 0]  # The estimate is the top left-singular vector of w.
    return estimate
