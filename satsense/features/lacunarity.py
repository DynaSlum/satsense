"""Lacunarity feature implementation."""
import numpy as np
from numba import jit, prange

from satsense.generators import CellGenerator
from satsense.generators.cell_generator import super_cell
from . import Feature


# @jit("float64(boolean[:, :], int64)", nopython=True, parallel=True)
@jit("float64(boolean[:, :], int64)", nopython=True)
def lacunarity(edged_image, box_size):
    """
    Calculate the lacunarity value over an image, following these papers:

    Kit, Oleksandr, and Matthias Lüdeke. "Automated detection of slum area change in Hyderabad, India using multitemporal satellite imagery." ISPRS journal of photogrammetry and remote sensing 83 (2013): 130-137.

    Kit, Oleksandr, Matthias Lüdeke, and Diana Reckien. "Texture-based identification of urban slums in Hyderabad, India using remote sensing data." Applied Geography 32.2 (2012): 660-667.
    """

    # accumulator holds the amount of ones for each position in the image, defined by a sliding window
    accumulator = np.zeros((edged_image.shape[0] - box_size,
                            edged_image.shape[1] - box_size))
    for i in prange(accumulator.shape[0]):
        for j in prange(accumulator.shape[1]):
            # sum the binary-box for the amount of 1s in this box
            accumulator[i, j] = np.sum(
                edged_image[i:i + box_size, j:j + box_size])

    accumulator = accumulator.flatten()
    mean_sqrd = np.mean(accumulator) ** 2
    if mean_sqrd == 0:
        return 0.0

    return (np.var(accumulator) / mean_sqrd) + 1


@jit
def lacunarity_for_chunk(chunk, box_sizes):
    len_box_sizes = len(box_sizes)
    chunk_len = len(chunk)

    chunk_matrix = np.zeros((chunk_len, len_box_sizes))
    coords = np.zeros((chunk_len, 2))
    for i in range(0, chunk_len):
        # Set coordinates
        coords[i, :] = chunk[i][0:2]

        # feature_vector = np.zeros(len(scales) * len(box_sizes))
        x, y, edged = chunk[i]
        for j in range(len_box_sizes):
            box_size = box_sizes[j]
            chunk_matrix[i, j] = lacunarity(edged, box_size)

    return coords, chunk_matrix


class Lacunarity(Feature):
    """Lacunarity feature."""

    def __init__(self, windows=((25, 25),), box_sizes=(10, 20, 30)):
        # Check input
        for window in windows:
            for box_size in box_sizes:
                if window[0] <= box_size or window[1] <= box_size:
                    raise ValueError(
                        "box_size {} must be smaller than window {}".format(
                            box_size, window))

        super(Lacunarity, self)
        self.box_sizes = box_sizes
        self.windows = windows
        self.feature_size = len(self.windows) * len(box_sizes)

    def __call__(self, cell):
        return lacunarity_for_chunk(cell, self.box_sizes)

    def initialize(self, generator: CellGenerator, scale):
        # Load the canny edged image for the whole image
        # This so it is not done on a window by window basis...
        sat_image = generator.image
        norm = sat_image.normalized
        ce = sat_image.canny_edged

        data = []
        for window in generator:
            edged, _, _ = super_cell(generator.image.canny_edged, scale, window.x_range, window.y_range, padding=True)
            processing_tuple = (window.x, window.y, edged)
            data.append(processing_tuple)

        return data

    def __str__(self):
        return "La-{}-{}".format(str(self.windows), str(self.box_sizes))
