"""Texton feature implementation."""
from functools import lru_cache
from typing import Iterator

import numpy as np
from scipy import ndimage
from skimage.filters import gabor_kernel, gaussian
from sklearn.cluster import MiniBatchKMeans

from satsense import SatelliteImage
from satsense.generators import CellGenerator
from satsense.generators.cell_generator import super_cell
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


@lru_cache(maxsize=1)
def create_kernels():
    """Create filter bank kernels."""
    kernels = []
    angles = 8
    thetas = np.linspace(0, np.pi, angles)
    for theta in thetas:
        # theta = theta / 8. * np.pi
        for sigma in (1,):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)

    return kernels


def get_texton_descriptors(image):
    """Compute texton descriptors."""
    kernels = create_kernels()
    length = len(kernels) + 1
    descriptors = np.zeros(image.shape + (length,), dtype=np.double)
    for k, kernel in enumerate(kernels):
        descriptors[:, :, k] = ndimage.convolve(image, kernel, mode='wrap')

    # Calculate Difference-of-Gaussian
    dog = gaussian(image, sigma=1) - gaussian(image, sigma=3)
    descriptors[:, :, length - 1] = dog

    return descriptors


def texton_for_chunk(chunk, kmeans, normalized=True):
    chunk_len = len(chunk)
    cluster_count = kmeans.n_clusters

    coords = np.zeros((chunk_len, 2))
    chunk_matrix = np.zeros((chunk_len, cluster_count))
    for i in range(chunk_len):
        coords[i, :] = chunk[i][0:2]

        im_grayscale = chunk[i][2]
        sub_descriptors = get_texton_descriptors(im_grayscale)

        # Calculate Difference-of-Gaussian
        dog = np.expand_dims(gaussian(im_grayscale, sigma=1) - gaussian(im_grayscale, sigma=3), axis=2)
        sub_descriptors = np.append(sub_descriptors, dog, axis=2)
        sub_descriptors = sub_descriptors.reshape(
            (sub_descriptors.shape[0] * sub_descriptors.shape[1], sub_descriptors.shape[2]))

        codewords = kmeans.predict(sub_descriptors)
        counts = np.bincount(codewords, minlength=cluster_count)

        # Perform normalization
        if normalized:
            counts = counts / cluster_count

        chunk_matrix[i, :] = counts

    return coords, chunk_matrix


class Texton(Feature):
    """Texton feature."""

    def __init__(self, kmeans: MiniBatchKMeans, windows=((25, 25),), normalized=True):
        super(Texton, self)
        self.windows = windows
        self.kmeans = kmeans
        self.feature_size = len(self.windows) * kmeans.n_clusters
        self.normalized = normalized
        self.descriptors = None

    def __call__(self, chunk):
        return texton_for_chunk(chunk, self.kmeans, self.normalized)

    def __str__(self):
        normalized = "n" if self.normalized == True else "nn"
        return "Te-{}-{}".format(str(self.windows), normalized)

    def chunk_size(self, cpu_cnt, im_shape):
        return im_shape[0]

    def initialize(self, generator: CellGenerator, scale):
        im_grayscale = generator.image.grayscale

        data = []
        for window in generator:
            im, x_range, y_range = super_cell(im_grayscale, scale, window.x_range, window.y_range, padding=True)
            processing_tuple = (window.x, window.y, im)
            data.append(processing_tuple)

        return data
