"""Texton feature implementation."""
import logging
from typing import Iterator

import numpy as np
from scipy.signal import convolve
from skimage.filters import gabor_kernel, gaussian
from sklearn.cluster import MiniBatchKMeans

from ..generators import FullGenerator
from ..image import Image
from .feature import Feature

logger = logging.getLogger(__name__)


def create_texton_kernels():
    """Create filter bank kernels."""
    kernels = []
    angles = 8
    thetas = np.linspace(0, np.pi, angles)
    for theta in thetas:
        for sigma in (1, ):
            for frequency in (0.05, ):
                kernel = np.real(
                    gabor_kernel(
                        frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)

    return kernels


def get_texton_descriptors(image: Image):
    """Compute texton descriptors."""
    logger.debug("Computing texton descriptors")
    kernels = create_texton_kernels()

    # Prepare input image
    array = image['grayscale']
    mask = array.mask
    array = array.filled(fill_value=0)

    # Create result image
    shape = array.shape + (len(kernels) + 1, )
    result = np.ma.empty(shape, dtype=array.dtype)
    result.mask = np.zeros(result.shape, dtype=bool)

    for k, kernel in enumerate(kernels):
        result[:, :, k] = convolve(array, kernel, mode='same')
        result.mask[:, :, k] = mask

    result[:, :, -1] = gaussian(array, sigma=1) - gaussian(array, sigma=3)
    result.mask[:, :, -1] = mask

    logger.debug("Done computing texton descriptors")
    return result


Image.register('texton_descriptors', get_texton_descriptors)


def texton_cluster(images: Iterator[Image],
                   n_clusters=32,
                   max_samples=100000,
                   sample_window=(8192, 8192)) -> MiniBatchKMeans:
    """Compute texton clusters."""
    nfeatures = int(max_samples / len(images))
    descriptors = []
    for image in images:
        chunk = np.minimum(image.shape, sample_window)

        generator = FullGenerator(image, chunk)
        generator.load_image('texton_descriptors', (chunk, ))

        max_features_per_window = int(nfeatures / np.prod(generator.shape))

        img_descriptors = []
        for array in generator:
            array = array.reshape(-1, array.shape[-1])
            non_masked = ~array.mask.any(axis=-1)
            data = array.data[non_masked]
            if data.shape[0] > max_features_per_window:
                data = data[np.random.choice(
                    data.shape[0], max_features_per_window, replace=False)]
            img_descriptors.append(data)
        img_descriptors = np.vstack(img_descriptors)

        descriptors.append(img_descriptors)
    descriptors = np.vstack(descriptors)

    # Cluster the descriptors
    mbkmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42).fit(descriptors)

    return mbkmeans


def texton(descriptors, kmeans: MiniBatchKMeans, normalized=True):
    """Calculate the texton feature on the given window."""
    n_clusters = kmeans.n_clusters

    shape = descriptors.shape
    descriptors = descriptors.reshape(shape[0] * shape[1], shape[2])

    codewords = kmeans.predict(descriptors)
    counts = np.bincount(codewords, minlength=n_clusters)

    # Perform normalization
    if normalized:
        counts = counts / n_clusters

    return counts


class Texton(Feature):
    """Texton feature."""

    base_image = 'texton_descriptors'
    compute = staticmethod(texton)

    def __init__(self, windows, kmeans: MiniBatchKMeans, normalized=True):
        """Create Texton feature."""
        super().__init__(windows, kmeans=kmeans, normalized=normalized)
        self.size = kmeans.n_clusters

    @classmethod
    def from_images(cls,
                    windows,
                    images: Iterator[Image],
                    n_clusters=32,
                    max_samples=100000,
                    sample_window=(8192, 8192),
                    normalized=True):
        kmeans = texton_cluster(
            images, n_clusters, max_samples, sample_window=sample_window)
        return cls(windows, kmeans, normalized)
