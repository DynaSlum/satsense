import logging
import math
import pickle
import sys
from enum import Enum

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as nd
from scipy import signal as sg
from scipy.ndimage import zoom
from skimage.feature import peak_local_max

from pysal.esda.getisord import G_Local
from pysal.weights.Distance import DistanceBand
from satsense.image import SatelliteImage

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Ktype(Enum):
    """
    This enum contains the different versions of the convolution kernels.

    """
    ORIGINAL = 1
    GAUSSIAN = 2
    INCREASE = 3
    NEGATIVE = 4


class Kernel:
    """
    This class produces a kernel that can be used for the detection of road
    intersections.

    Args:
        road_width:     The width of the road in the kernel in pixels; integer
        road_length:    The length of the road in the kernel in pixels;
                        integer
        kernel_type:    The type of kernel to be used. Available types are
                        listed in the Ktype enum; integer

    """

    def __init__(self,
                 road_width=30,
                 road_length=70,
                 kernel_type=Ktype.GAUSSIAN):
        self._road_width = road_width
        self._road_length = road_length
        self._kernel_type = kernel_type
        self._kernel = None

    def get(self):
        """
        Getter function for the convolution kernel.

        Returns:
            A kernel containing the shape of a cross; nxn numpy matrix

        """
        if self._kernel is None:
            self._kernel = self.__create()
        return self._kernel

    def __create(self):
        """
        This function is the parent function in the creation of convolution
        kernels. The kernel contains the form of cross to represent the form
        of an road intersection as seen from satellite images.

        """
        if self._kernel_type == Ktype.ORIGINAL:
            return self.__create_original_kernel()
        if self._kernel_type == Ktype.INCREASE:
            return self.__create_increase_kernel()
        if self._kernel_type == Ktype.NEGATIVE:
            return self.__create_negative_kernel()
        if self._kernel_type == Ktype.GAUSSIAN:
            return self.__create_gaussian_kernel()
        raise ValueError("Invalid kernel specified")

    def __create_original_kernel(self):
        """
        This function creates a type of kernel that was used as a proof of
        concept. The content of the kernel is a cross of ones with the
        remainder of the  kernel filled with zeros.

        Returns:
            A kernel containing the shape of a cross; nxn numpy matrix

        """
        # horizontal road
        hr = np.ones((self._road_width, self._road_length))
        # vertical road
        vr = np.ones((self._road_length, self._road_width))
        # road center
        cr = np.ones((self._road_width, self._road_width))
        # roadside
        rs = np.zeros((self._road_length, self._road_length))

        r1 = np.concatenate((rs, vr, rs), axis=1)
        r2 = np.concatenate((hr, cr, hr), axis=1)
        return np.concatenate((r1, r2, r1), axis=0)

    def __create_increase_kernel(self):
        """
        Creates a kernel where the ends of the intersection count the most.

        Returns:
            A kernel containing the shape of a cross; nxn numpy matrix

        """
        hr1 = np.tile(
            np.arange(self._road_length, 0, -1), (self._road_width, 1))
        hr2 = np.flip(hr1, axis=1)
        vr1 = np.transpose(hr1)
        vr2 = np.flip(vr1, axis=0)
        cr = np.ones((self._road_width, self._road_width))
        rs = np.zeros((self._road_length, self._road_length))

        max_val = 5
        r1 = np.concatenate((rs, vr1, rs), axis=1)
        r2 = np.concatenate((hr1, cr, hr2), axis=1)
        r3 = np.concatenate((rs, vr2, rs), axis=1)
        kernel = np.concatenate((r1, r2, r3), axis=0)
        kernel[kernel > max_val] = max_val
        return kernel

    def __create_negative_kernel(self):
        """
        Creates a kernel where the area outside the cross is negative.

        Returns:
            A kernel containing the shape of a cross; nxn numpy matrix

        """
        # horizontal road; all values are two
        hr = np.ones((self._road_width, self._road_length)) * 2
        # vertical road; all values are two
        vr = np.ones((self._road_length, self._road_width)) * 2
        # road center; all values are two
        cr = np.ones((self._road_width, self._road_width)) * 2

        min_val = -1
        # Create a staircase down from the cross to negative numbers. min_val
        # is lower bound of the negative numbers
        rs1 = np.stack([
            self.__calculate_row_negative_kernel(i, min_val)
            for i in range(1, self._road_length + 1)
        ])
        rs2 = np.flip(rs1, axis=1)
        rs3 = np.flip(rs1, axis=0)
        rs4 = np.flip(rs2, axis=0)

        r1 = np.concatenate((rs4, vr, rs3), axis=1)
        r2 = np.concatenate((hr, cr, hr), axis=1)
        r3 = np.concatenate((rs2, vr, rs1), axis=1)

        kernel = np.concatenate((r1, r2, r3), axis=0)
        kernel[kernel < min_val] = min_val
        return kernel

    def __calculate_row_negative_kernel(self, i, min_val):
        """
        A helper function for the negative kernel.
        """
        return np.concatenate((np.arange(-1, i * -1, -1),
                               np.full(self._road_length - i + 1, i * -1)))

    def __create_gaussian_kernel(self):
        """
        Creates a kernel where the cross of the kernel is built using two
        Gaussian distributions. The use of this distribution should create
        smoother results than the other kernels.

        Returns:
            A kernel containing the shape of a cross; nxn numpy matrix

        """
        kernel_width = self._road_length * 2 + self._road_width
        g1 = sg.gaussian(kernel_width, std=self._road_width / 2)

        r1 = np.tile(g1, (kernel_width, 1))
        r2 = np.transpose(r1)

        kernel = np.maximum(r1, r2)
        return kernel

    def __rotate_kernel(self, kernel, degrees):
        return nd.rotate(kernel, degrees)


class RoadIntersections:
    """
    This class detects road intersections in images.

    Args:
        image_path:         The path to the image to extract road intersections
                            from; string
        kernel:             Kernel object to use for convolution; Kernel object
        peak_min_distance:  The minimum distance between local maxima for
                            intersection detection; integer

    """

    def __init__(self, image, kernel, peak_min_distance=150):
        self._peak_min_distance = peak_min_distance
        self._image = image
        self._kernel = kernel
        self._intersections = None

    def get(self):
        """
        Getter function for the intersections

        Returns:
            A list of coordinates of the detected intersections on the image;
            nx2 numpy array

        """
        if self._intersections is None:
            self._intersections = self.__calculate()
        return self._intersections

    def visualize(self):
        """
        This functions displays the detected intersections on top of the input
        image.

        """
        if self._intersections is None:
            self._intersections = self.__calculate()

        plt.imshow(self._image.rgb)
        plt.scatter(
            self._intersections[:, 1],
            self._intersections[:, 0],
            c='r',
            alpha=0.5)
        plt.axis('off')
        plt.show()

    def __calculate(self):
        """
        This function uses convolution as a method for finding road
        intersections in an image.

        """
        gray_image = cv2.cvtColor(self._image.rgb, cv2.COLOR_BGR2GRAY)
        # Gives an error somehow
        # gray_image = cv2.threshold(
        #     gray_image,
        #     0,
        #     1,
        #     cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        # )[1]
        kernel = self._kernel.get()
        convolution = sg.convolve(gray_image, kernel, "valid")
        peaks = peak_local_max(
            convolution, min_distance=self._peak_min_distance)

        return self.__relocate_peaks(peaks)

    def __relocate_peaks(self, peaks):
        """
        This function relocates the peaks by the half of the kernel width.
        During the convolution, the kernel translates the image by the half of
        the kernel width. This relocation is necessary to move the peaks back
        to the right positions.

        """
        kernel_width = self._kernel.get().shape[0]
        return peaks + kernel_width / 2


class RoadIntersectionDensity:
    """
    This class represents the road intersection feature
    """

    def __init__(self, image, block_size=20, scale=150):
        self._image = image
        self._block_size = block_size
        self._scale = scale
        self._feature = None

    def get(self):
        """
        This function can be used to get the feature after the creation of the
        object

        """
        if self._feature is None:
            raise Exception("Feature is not yet created, run create() first.")

        return self._feature

    def create(self, intersections):
        """
        This function calculates the road intersection feature. It gets called
        automatically on the creation of this class.

        """
        density_map = self.__create_density_map(intersections.get())
        radius = int(self._scale / self._block_size)
        feature = self.__create_hotspot_map(density_map, radius)
        feature = self.__interpolate_feature(feature)
        self._feature = feature

    def visualize(self):
        if self._feature is None:
            raise Exception(
                "Feature not yet calculated, please run create() or load a "
                "feature using load()")

        plt.imshow(self._feature)
        plt.show()

    def save(self, path):
        LOG.info("Saving RID feature as: %s", path)
        f = open(path, 'w')
        if self._feature is not None:
            pickle.dump(self._feature, f)
        else:
            LOG.warning("RID feature was not yet calculated on save")

    def load(self, path):
        LOG.info("Opening RID feature file: %s", path)
        f = open(path, 'r')
        self._feature = pickle.load(f)

    def __create_density_map(self, points):
        """
        This function rasterizes the intersection points to a grid built from
        blocks of size block_size and in the shape of the image. This is
        required in the creation of a hotspot map from the intersection points.

        Args:
            points:         nx2 numpy array of integers containing the points
                            of road intersection.
            image_shape:    The shape of the input image; tuple of integers

        Returns:
            A rasterized version of the intersection points; nxm numpy matrix

        """
        height = self._image.shape[0]
        width = self._image.shape[1]
        scaled_block_size = self._block_size * 4

        density_map = np.zeros(
            (int(math.floor(float(height) / scaled_block_size)),
             int(math.floor(float(width) / scaled_block_size))))

        for point in points:
            h = int(point[0] / scaled_block_size)
            w = int(point[1] / scaled_block_size)

            if h < density_map.shape[0] and w < density_map.shape[1]:
                density_map[h, w] += 1
        return density_map

    def __create_hotspot_map(self, density_map, radius):
        """
        Create a hotspot map from the intersection density map.

        """
        grid = np.indices((density_map.shape[0], density_map.shape[1]))
        grid = np.stack((grid[0], grid[1]), axis=-1)
        grid = np.reshape(grid, (grid.shape[0] * grid.shape[1], 2))

        w = DistanceBand(grid, threshold=radius)
        y = np.ravel(density_map)

        g = G_Local(y, w).Zs
        return np.reshape(g, (density_map.shape[0], density_map.shape[1]))

    def __interpolate_feature(self, feature):
        """
        This function resizes and interpolates the feature matrix to the
        dimensions corresponding to the image with the correct block size. A
        larger blocksize was used to create the feature matrix to reduce the
        computational load.

        Args:
            feature:        The hotspot map of reduced dimensions; nxm numpy
                            matrix of floats
            image_shape:    The shape of the input image; tuple of integers
        Returns:
            A resized and interpolated version of the feature matrix in the
            correct dimensions corresponding to the shape of the image and
            blocksize.

        """
        feature_shape = feature.shape
        zoom_level = [
            self._image.shape[0] / (self._block_size * feature_shape[0]),
            self._image.shape[1] / (self._block_size * feature_shape[1]),
        ]

        # For the scipy UserWarning:
        # To compensate for the round() used in the zoom() when we want to use
        # a ceil() instead. The round() will give one off errors when the
        # computed dimensions of the interpolated feature matrix has the first
        # decimal lower than 0.5.
        if (zoom_level[0] * feature_shape[0]) % 1 < 0.5:
            zoom_level[0] = (
                math.ceil(zoom_level[0] * feature_shape[0]) / feature_shape[0])
        if (zoom_level[1] * feature_shape[1]) % 1 < 0.5:
            zoom_level[1] = (
                math.ceil(zoom_level[1] * feature_shape[1]) / feature_shape[1])

        return zoom(feature, zoom_level, order=3)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception("Please supply an image")

    image = SatelliteImage.load_from_file(sys.argv[1], 'worldview3')

    kernel = Kernel(road_width=15, road_length=50, kernel_type=Ktype.GAUSSIAN)
    intersections = RoadIntersections(image, kernel, peak_min_distance=100)
    rid = RoadIntersectionDensity(image, scale=80, block_size=30)
    rid.create(intersections)
    intersections.visualize()
    rid.visualize()
