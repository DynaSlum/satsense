"""Sift feature."""
from typing import Iterator

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from ..image import Image
from ..generators import FullGenerator
from .feature import Feature

def sift_cluster(images: Iterator[Image], n_clusters=32,
                 sample_size=100000, sample_window=(8192, 8192)) -> MiniBatchKMeans:
    """Create the clusters needed to compute the sift feature."""
    nfeatures = int(sample_size / len(images))
    SIFT = cv2.xfeatures2d.SIFT_create(nfeatures)

    descriptors = None
    for image in images:       
        if image.shape[0] < sample_window[0]:
            sample_window = (image.shape[0], sample_window[1])
        if image.shape[1] < sample_window[1]:
            sample_window = (sample_window[0], image.shape[1])

        generator = FullGenerator(image, sample_window)
        generator.load_image('gray_ubyte', (sample_window, ))

        for img in generator:
            inverse_mask = (~img.mask).astype(np.uint8)
            _, new_descriptors = SIFT.detectAndCompute(img, inverse_mask)
            del _  # Free up memory

            # Add descriptors if we already had some
            if descriptors is None:
                descriptors = new_descriptors
            else:
                descriptors = np.append(descriptors, new_descriptors, axis=0)

            if descriptors.shape[0] > nfeatures:
                # Limit the number of descriptors to nfeatures
                # by randomly selecting some rows
                descriptors = descriptors[np.random.choice(
                    descriptors.shape[0], nfeatures, replace=False), :]
                break

    # Cluster the descriptors
    mbkmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42).fit(descriptors)

    return mbkmeans


def sift(window_gray_ubyte, kmeans: MiniBatchKMeans, normalized=True):
    """Calculate the sift feature on the given window."""
    SIFT = cv2.xfeatures2d.SIFT_create()
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
        kmeans = sift_cluster(images, n_clusters, sample_size, sample_window=sample_window)
        return cls(windows, kmeans, normalized)
