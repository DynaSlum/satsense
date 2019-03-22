"""Lacunarity feature implementation."""
import logging

import numpy as np
import scipy
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
    gray_ubyte = image['gray_ubyte']
    mask = gray_ubyte.mask
    inverse_mask = ~mask
    result = equalize(gray_ubyte.data, selem=disk(radius), mask=inverse_mask)
    try:
        result = canny(result, sigma=sigma, mask=inverse_mask)
    except TypeError:
        logger.warning("Canny type error")
        result[:] = 0
    logger.debug("Done computing Canny edge image")
    return np.ma.array(result, mask=mask)


Image.register('canny_edge', get_canny_edge_image)


def lacunarity(edged_image, box_size):
    """
    Calculate the lacunarity value over an image.

    The calculation is performed following these papers:

    Kit, Oleksandr, and Matthias Luedeke. "Automated detection of slum area
    change in Hyderabad, India using multitemporal satellite imagery."
    ISPRS journal of photogrammetry and remote sensing 83 (2013): 130-137.

    Kit, Oleksandr, Matthias Luedeke, and Diana Reckien. "Texture-based
    identification of urban slums in Hyderabad, India using remote sensing
    data." Applied Geography 32.2 (2012): 660-667.
    """
    kernel = np.ones((box_size, box_size))
    accumulator = scipy.signal.convolve2d(edged_image, kernel, mode='valid')
    mean_sqrd = np.mean(accumulator)**2
    if mean_sqrd == 0:
        return 0.0

    return np.var(accumulator) / mean_sqrd + 1


def lacunarities(canny_edge_image, box_sizes):
    """Calculate the lacunarities for all box_sizes."""
    result = [lacunarity(canny_edge_image, box_size) for box_size in box_sizes]
    return result


class Lacunarity(Feature):
    """
    Calculate the lacunarity value over an image.

    Lacunarity is a measure of 'gappiness' of the image.
    The calculation is performed following these papers:

    Kit, Oleksandr, and Matthias Luedeke. "Automated detection of slum area
    change in Hyderabad, India using multitemporal satellite imagery."
    ISPRS journal of photogrammetry and remote sensing 83 (2013): 130-137.

    Kit, Oleksandr, Matthias Luedeke, and Diana Reckien. "Texture-based
    identification of urban slums in Hyderabad, India using remote sensing
    data." Applied Geography 32.2 (2012): 660-667.
    """

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
