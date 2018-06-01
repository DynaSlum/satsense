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
    descriptors = None

    for sat_image in sat_images:
        new_descriptors = get_texton_descriptors(sat_image.grayscale)
        shape = new_descriptors.shape
        new_descriptors.shape = (shape[0] * shape[1], shape[2])

        # Add descriptors if we already had some
        if descriptors is None:
            descriptors = new_descriptors
        else:
            descriptors = np.append(descriptors, new_descriptors, axis=0)

    # Sample {sample_size} descriptors from all descriptors
    # (Takes random rows) and cluster these
    print("Sampling from {}".format(descriptors.shape))
    descriptors = descriptors[np.random.choice(
        descriptors.shape[0], sample_size, replace=False), :]

    mbkmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42).fit(descriptors)

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
    print("Computing texton descriptors for image with shape {}"
          .format(image.shape))
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
        self.descriptors = None

    def __call__(self, cell):
        result = np.zeros(self.feature_size)
        n_clusters = self.kmeans.n_clusters
        for i, window in enumerate(self.windows):
            win = cell.super_cell(window, padding=True)
            start_index = i * n_clusters
            end_index = (i + 1) * n_clusters
            result[start_index:end_index] = self.texton(win)
        return result

    def __str__(self):
        normalized = "normalized" if self.normalized else "not-normalized"
        return "Texton1-dog-{}-{}".format(str(self.windows), normalized)

    def initialize(self, sat_image: SatelliteImage):
        """Compute texton descriptors for the image."""
        self.descriptors = get_texton_descriptors(sat_image.grayscale)

    def texton(self, window: Cell):
        """Calculate the texton feature on the given window."""
        cluster_count = self.kmeans.n_clusters

        descriptors = self.descriptors[window.x_range, window.y_range, :]
        shape = descriptors.shape
        descriptors = descriptors.reshape(shape[0] * shape[1], shape[2])

        codewords = self.kmeans.predict(descriptors)
        counts = np.bincount(codewords, minlength=cluster_count)

        # Perform normalization
        if self.normalized:
            counts = counts / cluster_count

        return counts
