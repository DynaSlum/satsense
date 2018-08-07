"""Texton feature implementation."""
from typing import Iterator

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .. import SatelliteImage
from .feature import Feature


def texton_cluster(sat_images: Iterator[SatelliteImage],
                   n_clusters=32,
                   sample_size=100000) -> MiniBatchKMeans:
    """Compute texton clusters."""
    descriptors = np.vstack(s.texton_descriptors for s in sat_images)
    shape = descriptors.shape
    descriptors = descriptors.reshape(shape[0] * shape[1], shape[2])

    if descriptors.shape[0] > sample_size:
        # Limit the number of descriptors to sample_size
        # by randomly selecting some rows
        descriptors = descriptors[np.random.choice(
            descriptors.shape[0], sample_size, replace=False), :]

    # Cluster the descriptors
    mbkmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42).fit(descriptors)

    return mbkmeans


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
        self.base_image = 'texton_descriptors'

    def __call__(self, cell):
        result = np.zeros(self.feature_size)
        n_clusters = self.kmeans.n_clusters
        for i, window in enumerate(self.windows):
            win = cell.super_cell(window, padding=True)
            start_index = i * n_clusters
            end_index = (i + 1) * n_clusters
            result[start_index:end_index] = self.texton(win.texton_descriptors)
        return result

    def __str__(self):
        normalized = "normalized" if self.normalized else "not-normalized"
        return "Texton1-dog-{}-{}".format(str(self.windows), normalized)

    def texton(self, descriptors):
        """Calculate the texton feature on the given window."""
        cluster_count = self.kmeans.n_clusters

        shape = descriptors.shape
        descriptors = descriptors.reshape(shape[0] * shape[1], shape[2])

        codewords = self.kmeans.predict(descriptors)
        counts = np.bincount(codewords, minlength=cluster_count)

        # Perform normalization
        if self.normalized:
            counts = counts / cluster_count

        return counts
