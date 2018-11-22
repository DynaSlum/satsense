"""Sift feature."""
from typing import Iterator

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from ..image import Image
from .feature import Feature

SIFT = cv2.xfeatures2d.SIFT_create()


def sift_cluster(images: Iterator[Image], n_clusters=32,
                 sample_size=100000) -> MiniBatchKMeans:
    """Create the clusters needed to compute the sift feature."""
    descriptors = None
    for image in images:
        array = image['gray_ubyte']
        inverse_mask = (~array.mask).astype(np.uint8)
        _, new_descriptors = SIFT.detectAndCompute(array, inverse_mask)
        del _  # Free up memory

        # Add descriptors if we already had some
        if descriptors is None:
            descriptors = new_descriptors
        else:
            descriptors = np.append(descriptors, new_descriptors, axis=0)

    if descriptors.shape[0] > sample_size:
        # Limit the number of descriptors to sample_size
        # by randomly selecting some rows
        descriptors = descriptors[np.random.choice(
            descriptors.shape[0], sample_size, replace=False), :]

    # Cluster the descriptors
    mbkmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42).fit(descriptors)

    return mbkmeans


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
                    normalized=True):
        kmeans = sift_cluster(images, n_clusters, sample_size)
        return cls(windows, kmeans, normalized)
