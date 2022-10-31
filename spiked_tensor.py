import jax.numpy as np
from jax import random

from utils import sample_sphere, d_fold_tensor_product


class SpikedTensor:
    def __init__(
        self,
        lmbda,
        dim,
        order=4,
        seed=0,
    ) -> None:

        # Signal-to-noise parameter.
        self.lmbda = lmbda

        # Dimension.
        self.dim = dim

        # Tensor order.
        self.order = order

        # Fix random seed.
        key = random.PRNGKey(seed)

        # Generate a sample.
        self.spike, self.Y = SpikedTensor.generate_sample(
            key, self.lmbda, self.dim, self.order
        )

    @staticmethod
    def generate_sample(
        key, lmbda, n, d
    ) -> tuple[np.DeviceArray, np.DeviceArray, np.DeviceArray]:
        """Generates a sample of the form Y = lambda*x^{\otimes d} + W/sqrt(n)"""
        key, subkey = random.split(key)
        spike = sample_sphere(subkey, n)  # sampled uniformly from the sphere
        key, subkey = random.split(key)
        Y = random.normal(subkey, d * (n,))  # iid standard normal noise
        Y = lmbda * d_fold_tensor_product(spike, d) + Y / np.sqrt(n)

        return key, spike, Y
