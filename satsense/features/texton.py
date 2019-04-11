"""Texton feature implementation."""
import logging
import math
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


def get_texton_descriptor_pixel(array, kernels=create_texton_kernels()):
    """Compute texton descriptors for a single pixel."""
    logger.debug("Computing texton descriptors")

    # Prepare input image
    # array = image['grayscale']
    array = array.filled(fill_value=0)

    # Create result image
    shape = len(kernels) + 1
    result = np.ma.empty(shape, dtype=array.dtype)
    result.mask = False

    for k, kernel in enumerate(kernels):
        result[k] = convolve(array, kernel, mode='valid')[0][0]

    dog = gaussian(array, sigma=1) - gaussian(array, sigma=3)
    center = math.floor(array.shape[0] / 2), math.floor(array.shape[1] / 2)
    result[-1] = dog[center]

    logger.debug("Done computing texton descriptors")
    return result


def calculate_textons_for_image(image, kernels, mbkmeans, coordinates):
    # offset = np.max([math.floor(k.shape[0] / 2) for k in kernels])
    offset = np.max([math.floor(k.shape[0] / 2) for k in kernels])
    descriptors = []
    for i, j in coordinates:
        slice1 = slice(i - offset, i + offset + 1)
        slice2 = slice(j - offset, j + offset + 1)

        window = image['grayscale'][slice1, slice2]
        sample = get_texton_descriptor_pixel(window, kernels=kernels)

        descriptors.append(sample)

        if len(descriptors) >= 100:
            descriptors = np.vstack(descriptors)
            mbkmeans.partial_fit(descriptors)
            descriptors = []

    # Also take the last ones into account
    if len(descriptors) > 0:
        descriptors = np.vstack(descriptors)
        mbkmeans.partial_fit(descriptors)


def texton_cluster(images: Iterator[Image], n_clusters=32,
                   max_samples=100000) -> MiniBatchKMeans:
    """Compute texton clusters."""
    # Cluster the descriptors
    mbkmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    kernels = create_texton_kernels()

    rand_state = np.random.RandomState(seed=0)
    samples_per_image = int(max_samples / len(images))
    for image in images:
        offset = np.max([k.shape[0] for k in kernels])
        padding = (offset, offset)
        image.precompute_normalization()

        image = image.copy_block(((-padding[0], image.shape[0] + padding[0]),
                                  (-padding[1], image.shape[1] + padding[1])))
        old_cached = image.cached
        image.cached = ['grayscale']

        if np.prod(image.shape) < samples_per_image:
            x_range = padding[0] + np.array(range(0, image.shape[0]))
            y_range = padding[1] + np.array(range(0, image.shape[1]))
        else:
            x_range = padding[0] + rand_state.randint(0, image.shape[0],
                                                      samples_per_image)
            y_range = padding[1] + rand_state.randint(0, image.shape[0],
                                                      samples_per_image)

        calculate_textons_for_image(image, kernels, mbkmeans,
                                    zip(x_range, y_range))

        image.cached = old_cached

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
    """
    Texton Feature Transform calculator

    First create a codebook of Texton features from the suplied images using
    `from_images`. Then we can compute the histogram of codewords for a given
    window.

    For more information see [1]_.

    Parameters
    ----------
    window_shapes: list
        The window shapes to calculate the feature on.
    kmeans : sklearn.cluster.MiniBatchKMeans
        The trained KMeans clustering from opencv
    normalized : bool
        If True normalize the feature by the total number of clusters

    Example
    -------
    Calculating the Texton feature on an image using a generator::

        from satsense import Image
        from satsense.generators import FullGenerator
        from satsense.extract import extract_feature
        from satsense.features import Texton

        windows = ((50, 50), )

        image = Image('test/data/source/section_2_sentinel.tif', 'quickbird')
        image.precompute_normalization()

        texton = Texton.from_images(windows, [image])

        generator = FullGenerator(image, (10, 10))

        feature_vector = extract_feature(texton, generator)
        print(feature_vector.shape)

    Notes
    -----
    .. [1] Arbelaez, Pablo, et al., "Contour detection and hierarchical
           image segmentation," IEEE transactions on pattern analysis and
           machine intelligence (2011), vol. 33 no. 5, pp. 898-916.
    """

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
                    normalized=True):
        """
        Create a codebook of Texton features from the suplied images.

        Using the images `max_samples` Texton features are extracted
        evenly from all images. These features are then clustered into
        `n_clusters` clusters. This codebook can then be used to
        calculate a histogram of this codebook.

        Parameters
        ----------
        windows : list[tuple]
            The window shapes to calculate the feature on.
        images : Iterator[satsense.Image]
            Iterable for the images to calculate the codebook no
        n_cluster : int
            The number of clusters to create for the codebook
        max_samples : int
            The maximum number of samples to use for creating the codebook
        normalized : bool
            Wether or not to normalize the resulting feature with regards to
            the number of clusters
        """
        kmeans = texton_cluster(images, n_clusters, max_samples)
        return cls(windows, kmeans, normalized)
