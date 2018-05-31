"""Texton feature implementation."""
from typing import Iterator

import numpy as np
from scipy import ndimage
from skimage.filters import gabor_kernel, gaussian
from sklearn.cluster import MiniBatchKMeans

from .. import SatelliteImage
from ..generators.cell_generator import Cell
from .feature import Feature


def texton_cluster(sat_images: Iterator[SatelliteImage],
                   n_clusters=32,
                   sample_size=100000) -> MiniBatchKMeans:
    """Compute texton clusters."""
    base_descriptors = None

    for sat_image in sat_images:
        print("Computing texton descriptors for sat_image {}".format(
            str(sat_image.shape)))
        descriptors = get_texton_descriptors(sat_image.grayscale)
        shape = descriptors.shape
        descriptors.shape = (shape[0] * shape[1], shape[2])

        # Add descriptors if we already had some
        if base_descriptors is None:
            base_descriptors = descriptors
        else:
            base_descriptors = np.append(base_descriptors, descriptors, axis=0)

    # Sample {sample_size} descriptors from all descriptors
    # (Takes random rows) and cluster these
    print("Sampling from {}".format(str(base_descriptors.shape)))
    X = base_descriptors[np.random.choice(
        base_descriptors.shape[0], sample_size, replace=False), :]

    mbkmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42).fit(X)

    return mbkmeans


def create_kernels():
    """Create filter bank kernels."""
    kernels = []
    angles = 8
    thetas = np.linspace(0, np.pi, angles)
    for theta in thetas:
        # theta = theta / 8. * np.pi
        for sigma in (1, ):
            for frequency in (0.05, ):
                kernel = np.real(
                    gabor_kernel(
                        frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)

    return kernels


def get_texton_descriptors(image):
    """Compute texton descriptors."""
    kernels = create_kernels()
    length = len(kernels) + 1
    descriptors = np.zeros(image.shape + (length, ), dtype=np.double)
    for k, kernel in enumerate(kernels):
        descriptors[:, :, k] = ndimage.convolve(image, kernel, mode='wrap')

    # Calculate Difference-of-Gaussian
    dog = gaussian(image, sigma=1) - gaussian(image, sigma=3)
    descriptors[:, :, length - 1] = dog

    return descriptors


class Texton(Feature):
    """Texton feature."""
    def __init__(self,
                 kmeans: MiniBatchKMeans,
                 windows=((25, 25), ),
                 normalized=True):
        super(Texton, self)
        self.windows = windows
        self.kmeans = kmeans
        self.feature_size = len(self.windows) * kmeans.n_clusters
        self.normalized = normalized
        self.kernels = create_kernels()
        self.descriptors = None

    def __call__(self, cell):
        result = np.zeros(self.feature_size)
        n_clusters = self.kmeans.n_clusters
        for i, window in enumerate(self.windows):
            win = cell.super_cell(window, padding=True)
            start_index = i * n_clusters
            end_index = (i + 1) * n_clusters
            result[start_index:end_index] = self.texton(win, self.kmeans)
        return result

    def __str__(self):
        normalized = "normalized" if self.normalized else "not-normalized"
        return "Texton1-dog-{}-{}".format(str(self.windows), normalized)

    def initialize(self, sat_image: SatelliteImage):
        """Compute texton descriptors for the image."""
        self.descriptors = get_texton_descriptors(sat_image.grayscale)

    def texton(self, window: Cell, kmeans: MiniBatchKMeans):
        """
        Calculate the sift feature on the given window

        Args:
            window (nparray): A window of an image
            maximum (int): The maximum value in the image
        """
        cluster_count = kmeans.n_clusters

        descriptors = self.descriptors[window.x_range, window.y_range, :]
        shape = descriptors.shape
        descriptors = descriptors.reshape(shape[0] * shape[1], shape[2])

        codewords = kmeans.predict(descriptors)
        counts = np.bincount(codewords, minlength=cluster_count)

        # Perform normalization
        if self.normalized:
            counts = counts / cluster_count

        return counts
