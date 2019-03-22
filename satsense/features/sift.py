"""Sift feature."""
from typing import Iterator

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from ..generators import FullGenerator
from ..image import Image
from .feature import Feature


def sift_cluster(images: Iterator[Image],
                 n_clusters=32,
                 max_samples=100000,
                 sample_window=(8192, 8192)) -> MiniBatchKMeans:
    """Create the clusters needed to compute the sift feature."""
    nfeatures = int(max_samples / len(images))
    descriptors = []
    for image in images:
        image.precompute_normalization()

        chunk = np.minimum(image.shape, sample_window)

        generator = FullGenerator(image, chunk)
        generator.load_image('gray_ubyte', (chunk, ))

        max_features_per_window = int(nfeatures / np.prod(generator.shape))
        sift_object = cv2.xfeatures2d.SIFT_create(max_features_per_window)

        for img in generator:
            inverse_mask = (~img.mask).astype(np.uint8)
            new_descr = sift_object.detectAndCompute(img, inverse_mask)[1]
            descriptors.append(new_descr)

    descriptors = np.vstack(descriptors)

    # Cluster the descriptors
    mbkmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42).fit(descriptors)

    return mbkmeans


SIFT = cv2.xfeatures2d.SIFT_create()
"""SIFT feature calculator used by the sift function."""


def sift(window_gray_ubyte, kmeans: MiniBatchKMeans, normalized=True):
    """
    Calculate the Scale-Invariant Feature Transform feature

    The opencv SIFT features are first calculated on the window
    the codewords of these features are then extracted using the
    previously computed cluster centers. Finally a histogram of
    these codewords is returned

    Parameters
    ----------
    window_gray_ubyte : ndarray
        The window to calculate the feature on
    kmeans : sklearn.cluster.MiniBatchKMeans
        The trained KMeans clustering from opencv, see `from_images`
    normalized : bool
        If True normalize the feature by the total number of clusters

    Returns
    -------
        ndarray
            The histogram of sift feature codewords
    """
    descriptors = SIFT.detectAndCompute(window_gray_ubyte, None)[1]

    # Is none if no descriptors are found, i.e. on 0 input range
    n_clusters = kmeans.n_clusters
    if descriptors is None:
        return np.zeros(n_clusters)

    codewords = kmeans.predict(descriptors)
    counts = np.bincount(codewords, minlength=n_clusters)

    # Perform normalization
    if normalized:
        counts = counts / n_clusters

    return counts


class Sift(Feature):
    """
    Scale-Invariant Feature Transform calculator

    First create a codebook of SIFT features from the suplied images using
    `from_images`. Then we can compute the histogram of codewords for a given
    window.

    See the opencv
    `SIFT intro
    <https://docs.opencv.org/3.4.3/da/df5/tutorial_py_sift_intro.html>`__
    for more information

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
    Calculating the Sift feature on an image using a generator::

        from satsense import Image
        from satsense.generators import FullGenerator
        from satsense.extract import extract_feature
        from satsense.features import Sift

        windows = ((50, 50), )

        image = Image('test/data/source/section_2_sentinel.tif', 'quickbird')
        image.precompute_normalization()

        sift = Sift.from_images(windows, [image])

        generator = FullGenerator(image, (10, 10))

        feature_vector = extract_feature(sift, generator)
        print(feature_vector.shape)

    """

    base_image = 'gray_ubyte'
    compute = staticmethod(sift)

    def __init__(self, windows, kmeans: MiniBatchKMeans, normalized=True):
        """Create sift feature."""
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
        """
        Create a codebook of SIFT features from the suplied images.

        Using the images `max_samples` SIFT features are extracted
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
        kmeans = sift_cluster(
            images, n_clusters, max_samples, sample_window=sample_window)
        return cls(windows, kmeans, normalized)
