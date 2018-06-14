"""Sift feature."""
from typing import Iterator

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

from .. import SatelliteImage
from .feature import Feature
from satsense.generators import CellGenerator
from satsense.generators.cell_generator import super_cell


def sift_cluster(sat_images: Iterator[SatelliteImage],
                 n_clusters=32,
                 sample_size=100000) -> MiniBatchKMeans:
    """Create the clusters needed to compute the sift feature."""
    sift = cv2.xfeatures2d.SIFT_create()
    descriptors = None
    for sat_image in sat_images:
        _, new_descriptors = sift.detectAndCompute(sat_image.gray_ubyte, None)
        del _  # Free up memory

        # Add descriptors if we already had some
        if descriptors is None:
            descriptors = new_descriptors
        else:
            descriptors = np.append(descriptors, new_descriptors, axis=0)

    # Sample {sample_size} descriptors from all descriptors
    # (Takes random rows) and cluster these
    descriptors = descriptors[np.random.choice(
        descriptors.shape[0], sample_size, replace=False), :]

    mbkmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=42).fit(descriptors)

    return mbkmeans


def sift_for_chunk(chunk, kmeans, normalized=True):
    chunk_len = len(chunk)
    cluster_count = kmeans.n_clusters
    sift_obj = cv2.xfeatures2d.SIFT_create()

    coords = np.zeros((chunk_len, 2))
    chunk_matrix = np.zeros((chunk_len, cluster_count), dtype=np.float64)
    for i in range(chunk_len):
        coords[i, :] = chunk[i][0:2]

        win_gray_ubyte = chunk[i][2]

        kp, descriptors = sift_obj.detectAndCompute(win_gray_ubyte, None)
        del kp  # Free up memory

        if descriptors is None:
            chunk_matrix[i, :] = np.zeros((cluster_count), dtype=np.int32)
            continue

        codewords = kmeans.predict(descriptors)
        counts = np.bincount(codewords, minlength=cluster_count)

        # Perform normalization
        if normalized:
            counts = counts / cluster_count

        chunk_matrix[i, :] = counts

    return coords, chunk_matrix


class Sift(Feature):
    """Sift feature."""

    def __init__(self,
                 kmeans: MiniBatchKMeans,
                 windows=((25, 25),),
                 normalized=True):
        """Create sift feature."""
        super(Sift, self)
        self.windows = windows
        self.kmeans = kmeans
        self.feature_size = len(self.windows) * kmeans.n_clusters
        self.sift_obj = cv2.xfeatures2d.SIFT_create()
        self.normalized = normalized

    def __call__(self, chunk):
        return sift_for_chunk(chunk, self.kmeans, self.normalized)

    def __str__(self):
        normalized = "n" if self.normalized == True else "nn"
        return "Si-{}{}".format(str(self.windows), normalized)

    def initialize(self, generator: CellGenerator, scale):
        data = []
        for window in generator:
            win_gray_ubyte, _, _ = super_cell(generator.image.gray_ubyte, scale, window.x_range, window.y_range,
                                              padding=False)
            processing_tuple = (window.x, window.y, win_gray_ubyte)
            data.append(processing_tuple)

        return data
