import jax.numpy as np
from jax.numpy.linalg import svd


class RecursiveUnfolding:
    def __init__(self, Y) -> None:
        self.Y = Y

    def run(self) -> np.DeviceArray:
        """
        Estimate the spike according to recursive tensor unfolding.
        (Sec 3.2 in https://arxiv.org/abs/1411.1076.)

        Args:
            Y (np.DeviceArray): Spiked tensor sample.

        Returns:
            np.DeviceArray: Estimated spike.
        """

        """
        Antoine: The algorithm is valid for d odd as well?
        F: Yes, that's why we use ceil and floor. There's no theoretical
        guarantee for performance, but see e.g. sec 6.2, fig. 2. There
        they use d=3.
        """
        n = self.Y.shape[0]  # Dimension.
        d = len(self.Y.shape)  # Tensor order.
        Mat_Y = self.Y.reshape(n ** np.floor(d / 2), n ** np.ceil(d / 2))
        *_, Vh = svd(Mat_Y, full_matrices=False)
        w = Vh[0]  # w is the top right-singular vector of Mat_Y.
        # TODO: # Antoine: cf the footnote of Sec 3.2 of https://arxiv.org/pdf/1411.1076.pdf
        # (page 10): A better way might be to create a more "balanced" matrix from W, take
        # its top eigenvector and then iterate. It may not be necessary but we should keep
        # it in mind in case we have a computation time bottleneck for this algorithm.
        Mat_w = w.reshape(n, n ** (np.ceil(d / 2) - 1))
        U, *_ = svd(Mat_w, full_matrices=False)
        self.estimate = U[:, 0]  # The estimate is the top left-singular vector of w.
