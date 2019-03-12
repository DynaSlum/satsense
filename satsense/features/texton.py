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


def texton_cluster(images: Iterator[Image], n_clusters=32,
                   sample_size=100000,
                   sample_window=(8192, 8192)) -> MiniBatchKMeans:
    """Compute texton clusters."""
    nfeatures = int(sample_size / len(images))
    descriptors = []
    for image in images:       
        if image.shape[0] < sample_window[0]:
            sample_window = (image.shape[0], sample_window[1])
        if image.shape[1] < sample_window[1]:
            sample_window = (sample_window[0], image.shape[1])

        generator = FullGenerator(image, sample_window)
        generator.load_image('texton_descriptors', (sample_window, ))

        img_descriptors = []
        for img in generator:
            array = img
            array = array.reshape(-1, array.shape[-1])
            non_masked = ~array.mask.any(axis=-1)
            img_descriptors.append(array.data[non_masked])
        img_descriptors = np.vstack(img_descriptors)

        if img_descriptors.shape[0] > nfeatures:
            # Limit the number of descriptors to nfeatures
            # by randomly selecting some rows
            img_descriptors = img_descriptors[np.random.choice(
                img_descriptors.shape[0], nfeatures, replace=False), :]

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
                    sample_size=100000,
                    sample_window=(8192, 8192),
                    normalized=True):
        kmeans = texton_cluster(images, n_clusters, sample_size, sample_window=sample_window)
        return cls(windows, kmeans, normalized)
