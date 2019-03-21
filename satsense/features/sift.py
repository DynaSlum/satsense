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
        chunk = np.minimum(image.shape, sample_window)

        generator = FullGenerator(image, chunk)
        generator.load_image('gray_ubyte', (chunk, ))

        max_features_per_window = int(nfeatures / np.prod(generator.shape))
        sift_object = cv2.xfeatures2d.SIFT_create(max_features_per_window)

        for img in generator:
            inverse_mask = (~img.mask).astype(np.uint8)
            _, new_descriptors = sift_object.detectAndCompute(
                img, inverse_mask)
            descriptors.append(new_descriptors)

    descriptors = np.vstack(descriptors)

    # Cluster the descriptors
    mbkmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42).fit(descriptors)

    return mbkmeans


SIFT = cv2.xfeatures2d.SIFT_create()
"""SIFT feature calculator used by the sift function."""


def sift(window_gray_ubyte, kmeans: MiniBatchKMeans, normalized=True):
    """Calculate the sift feature on the given window."""
    _, descriptors = SIFT.detectAndCompute(window_gray_ubyte, None)
    del _  # Free up memory

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
    """Sift feature."""

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
                    sample_size=100000,
                    sample_window=(8192, 8192),
                    normalized=True):
        kmeans = sift_cluster(
            images, n_clusters, sample_size, sample_window=sample_window)
        return cls(windows, kmeans, normalized)
