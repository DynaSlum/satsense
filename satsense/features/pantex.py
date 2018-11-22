import numpy as np
import scipy as sp
from skimage.feature import greycomatrix, greycoprops

from .feature import Feature


def get_rii_dist_angles():
    """
    Get the angles and distances of the pixels used in the paper:

    Graesser, Jordan, et al. "Image based characterization of formal and informal
    neighborhoods in an urban landscape." IEEE Journal of Selected Topics in
    Applied Earth Observations and Remote Sensing 5.4 (2012): 1164-1176.
    """
    # Result caching
    if not hasattr(get_rii_dist_angles, "offsets"):
        pixels_dist_14 = np.array([[1, -1], [1, 1]])
        pixels_dist_1 = np.array([[0, 1], [1, 0]])
        pixels_dist_2 = np.array([[2, 0], [0, 2]])
        pixels_dist_223 = np.array([[1, -2], [2, -1], [2, 1], [1, 2]])
        pixels_dist_3 = np.array([[2, 2], [2, -2]])

        angles_1 = np.arctan2(pixels_dist_1[:, 0], pixels_dist_1[:, 1])
        distances_1 = [
            sp.spatial.distance.euclidean(
                [0, 0], [pixels_dist_1[i, 0], pixels_dist_1[i, 1]])
            for i in range(len(pixels_dist_1[:, 0]))
        ]

        angles_14 = np.arctan2(pixels_dist_14[:, 0], pixels_dist_14[:, 1])
        distances_14 = [
            sp.spatial.distance.euclidean(
                [0, 0], [pixels_dist_14[i, 0], pixels_dist_14[i, 1]])
            for i in range(len(pixels_dist_14[:, 0]))
        ]

        angles_2 = np.arctan2(pixels_dist_2[:, 0], pixels_dist_2[:, 1])
        distances_2 = [
            sp.spatial.distance.euclidean(
                [0, 0], [pixels_dist_2[i, 0], pixels_dist_2[i, 1]])
            for i in range(len(pixels_dist_2[:, 0]))
        ]

        angles_223 = np.arctan2(pixels_dist_223[:, 0], pixels_dist_223[:, 1])
        distances_223 = [
            sp.spatial.distance.euclidean(
                [0, 0], [pixels_dist_223[i, 0], pixels_dist_223[i, 1]])
            for i in range(len(pixels_dist_223[:, 0]))
        ]

        angles_3 = np.arctan2(pixels_dist_3[:, 0], pixels_dist_3[:, 1])
        distances_3 = [
            sp.spatial.distance.euclidean(
                [0, 0], [pixels_dist_3[i, 0], pixels_dist_3[i, 1]])
            for i in range(len(pixels_dist_3[:, 0]))
        ]

        offsets_1 = np.stack((distances_1, angles_1), axis=1)
        offsets_14 = np.stack((distances_14, angles_14), axis=1)
        offsets_2 = np.stack((distances_2, angles_2), axis=1)
        offsets_223 = np.stack((distances_223, angles_223), axis=1)
        offsets_3 = np.stack((distances_3, angles_3), axis=1)
        offsets = np.concatenate((offsets_1, offsets_14, offsets_2,
                                  offsets_223, offsets_3))

        # Cache the results, this function is called often!
        get_rii_dist_angles.offsets = offsets

    # print(offsets.shape)
    # for i in range(len(offsets)):
    #     print("Distance: {}, angle: {}".format(offsets[i][0], offsets[i][1]))

    return get_rii_dist_angles.offsets


def pantex(window, maximum=255):
    """
    Calculate the pantex feature on the given grayscale window

    Args:
        window (nparray): A window of an image
        maximum (int): The maximum value in the image
    """
    offsets = get_rii_dist_angles()

    pan = np.zeros(len(offsets))
    for i, offset in enumerate(offsets):
        glcm = greycomatrix(
            window, [offset[0]], [offset[1]],
            symmetric=True,
            normed=True,
            levels=maximum + 1)
        pan[i] = greycoprops(glcm, 'contrast')

    return pan.min()


class Pantex(Feature):
    base_image = 'gray_ubyte'
    size = 1
    compute = staticmethod(pantex)
