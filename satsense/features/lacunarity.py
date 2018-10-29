"""Lacunarity feature implementation."""
import logging

import numpy as np
from numba import jit, prange
from skimage.feature import canny
from skimage.filters.rank import equalize
from skimage.morphology import disk

from . import Feature
from ..image import Image

logger = logging.getLogger(__name__)


def get_canny_edge_image(image: Image, radius=30, sigma=0.5):
    """Compute Canny edge image."""
    logger.debug("Computing Canny edge image")
    # local histogram equalization
    grayscale = equalize(image['grayscale'], selem=disk(radius))
    try:
        result = canny(grayscale, sigma=sigma)
    except TypeError:
        logger.warning("Canny type error")
        result = np.zeros(image.shape)
    logger.debug("Done computing Canny edge image")
    return result


Image.register('canny_edge', get_canny_edge_image)


@jit("float64(boolean[:, :], int64)", nopython=True)
def lacunarity(edged_image, box_size):
    """
    Calculate the lacunarity value over an image, following these papers:

    Kit, Oleksandr, and Matthias Luedeke. "Automated detection of slum area
    change in Hyderabad, India using multitemporal satellite imagery."
    ISPRS journal of photogrammetry and remote sensing 83 (2013): 130-137.

    Kit, Oleksandr, Matthias Luedeke, and Diana Reckien. "Texture-based
    identification of urban slums in Hyderabad, India using remote sensing
    data." Applied Geography 32.2 (2012): 660-667.
    """

    # accumulator holds the amount of ones for each position in the image,
    # defined by a sliding window
    accumulator = np.zeros((edged_image.shape[0] - box_size,
                            edged_image.shape[1] - box_size))
    for i in prange(accumulator.shape[0]):
        for j in prange(accumulator.shape[1]):
            # sum the binary-box for the amount of 1s in this box
            accumulator[i, j] = np.sum(
                edged_image[i:i + box_size, j:j + box_size])
    mean_sqrd = np.mean(accumulator)**2
    if mean_sqrd == 0:
        return 0.0

    return np.var(accumulator) / mean_sqrd + 1


@jit
def lacunarities(canny_edge_image, box_sizes):
    result = np.zeros(len(box_sizes))
    for i, box_size in enumerate(box_sizes):
        result[i] = lacunarity(canny_edge_image, box_size)
    return result


class Lacunarity(Feature):
    """Lacunarity feature."""
    base_image = 'canny_edge'
    compute = staticmethod(lacunarities)

    def __init__(self, windows=((25, 25), ), box_sizes=(10, 20, 30)):
        # Check input
        for window in windows:
            for box_size in box_sizes:
                if window[0] <= box_size or window[1] <= box_size:
                    raise ValueError(
                        "box_size {} must be smaller than window {}".format(
                            box_size, window))
        super().__init__(windows, box_sizes=box_sizes)
        self.size = len(box_sizes)
