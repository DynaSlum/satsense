from typing import Iterator

import numpy as np
from scipy import ndimage
from skimage.filters import gabor_kernel, gaussian
from sklearn.cluster import MiniBatchKMeans

from .. import SatelliteImage
from ..generators.cell_generator import Cell
from .feature import Feature


def texton_cluster(sat_images: Iterator[SatelliteImage],
                   n_clusters=32,
                   sample_size=100000) -> MiniBatchKMeans:
    base_descriptors = None

    # prepare filter bank kernels
    kernels = create_kernels()
    for sat_image in sat_images:
        print("Computing texton descriptors for sat_image {}".format(
            str(sat_image.shape)))
        descriptors = compute_feats(sat_image.grayscale, kernels)
        descriptors = descriptors.reshape(
            (descriptors.shape[0] * descriptors.shape[1],
             descriptors.shape[2]))

        # Compute DoG
        dog = np.expand_dims(
            (gaussian(sat_image.grayscale, sigma=1) - gaussian(
                sat_image.grayscale, sigma=3)).ravel(),
            axis=1)
        descriptors = np.append(descriptors, dog, axis=1)

        # Add descriptors if we already had some
        if base_descriptors is None:
            base_descriptors = descriptors
        else:
            base_descriptors = np.append(base_descriptors, descriptors, axis=0)

    # Sample {sample_size} descriptors from all descriptors
    # (Takes random rows) and cluster these
    print("Sampling from {}".format(str(base_descriptors.shape)))
    X = base_descriptors[np.random.choice(
        base_descriptors.shape[0], sample_size, replace=False), :]

    mbkmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42).fit(X)

    return mbkmeans


def create_kernels():
    kernels = []
    angles = 8
    thetas = np.linspace(0, np.pi, angles)
    for theta in thetas:
        # theta = theta / 8. * np.pi
        for sigma in (1, ):
            for frequency in (0.05, ):
                kernel = np.real(
                    gabor_kernel(
                        frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)

    return kernels


def compute_feats(image, kernels):
    feats = np.zeros(image.shape + (len(kernels), ), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = ndimage.convolve(image, kernel, mode='wrap')
        feats[:, :, k] = filtered

    return feats


class Texton(Feature):
    def __init__(self,
                 kmeans: MiniBatchKMeans,
                 windows=((25, 25), ),
                 normalized=True):
        super(Texton, self)
        self.windows = windows
        self.kmeans = kmeans
        self.feature_size = len(self.windows) * kmeans.n_clusters
        self.normalized = normalized
        self.kernels = create_kernels()
        self.descriptors = None

    def __call__(self, cell):
        result = np.zeros(self.feature_size)
        n_clusters = self.kmeans.n_clusters
        for i, window in enumerate(self.windows):
            win = cell.super_cell(window, padding=True)
            start_index = i * n_clusters
            end_index = (i + 1) * n_clusters
            result[start_index:end_index] = self.texton(win, self.kmeans)
        return result

    def __str__(self):
        normalized = "normalized" if self.normalized == True else "not-normalized"
        return "Texton1-dog-{}-{}".format(str(self.windows), normalized)

    def initialize(self, sat_image: SatelliteImage):
        im_grayscale = sat_image.grayscale

        descriptors = compute_feats(im_grayscale, self.kernels)

        # Calculate Difference-of-Gaussian
        dog = np.expand_dims(
            gaussian(im_grayscale, sigma=1) - gaussian(im_grayscale, sigma=3),
            axis=2)
        self.descriptors = np.append(descriptors, dog, axis=2)

    def texton(self, window: Cell, kmeans: MiniBatchKMeans):
        """
        Calculate the sift feature on the given window

        Args:
            window (nparray): A window of an image
            maximum (int): The maximum value in the image
        """
        # descriptors = compute_feats(window_gray_ubyte, self.kernels)
        # dog = np.expand_dims((gaussian(window_gray_ubyte, sigma=1) - gaussian(window_gray_ubyte, sigma=3)).ravel(), axis=1)
        # descriptors = np.append(descriptors, dog, axis=1)
        # if descriptors is None:
        #     return np.zeros((cluster_count))

        cluster_count = kmeans.n_clusters

        descriptors = self.descriptors[window.x_range, window.y_range, :]
        descriptors = descriptors.reshape(
            (descriptors.shape[0] * descriptors.shape[1],
             descriptors.shape[2]))

        codewords = kmeans.predict(descriptors)
        counts = np.bincount(codewords, minlength=cluster_count)

        # Perform normalization
        if self.normalized:
            counts = counts / cluster_count

        return counts
