import numpy as np
import scipy as sp

from satsense.util import load_from_file, normalize_image, get_rgb_image
from satsense.util import RGB, QUICKBIRD, WORLDVIEW2

from skimage import data, color, exposure, img_as_ubyte, img_as_uint
from skimage.feature import greycomatrix, greycoprops

def get_rii_dist_angles():
    """
    Get the angles and distances of the pixels used in the paper:
    """
    pixels_dist_1 = np.array([[1, -1], [1, 0], [1, 1], [0, 1]])
    pixels_dist_2 = np.array([[1, -2], [2, -1], [2, 0], [2, 1], [1, 2], [0, 2]])
    pixels_dist_3 = np.array([[2, 2], [2, -2]])

    angles_1 = np.arctan2(pixels_dist_1[:, 0], pixels_dist_1[:, 1])
    distances_1 = [sp.spatial.distance.euclidean([0, 0],
                                                 [pixels_dist_1[i, 0], pixels_dist_1[i, 1]])
                   for i in range(len(pixels_dist_1[:, 0]))]

    angles_2 = np.arctan2(pixels_dist_2[:, 0], pixels_dist_2[:, 1])
    distances_2 = [sp.spatial.distance.euclidean([0, 0],
                                                 [pixels_dist_2[i, 0], pixels_dist_2[i, 1]])
                   for i in range(len(pixels_dist_2[:, 0]))]

    angles_3 = np.arctan2(pixels_dist_3[:, 0], pixels_dist_3[:, 1])
    distances_3 = [sp.spatial.distance.euclidean([0, 0],
                                                 [pixels_dist_3[i, 0], pixels_dist_3[i, 1]])
                   for i in range(len(pixels_dist_3[:, 0]))]

    return (angles_1, angles_2, angles_3), (distances_1, distances_2, distances_3)

def glcm(image, bands, normalize=True,
         symmetric=False, normed=False):
    rgb_image = get_rgb_image(image, bands, normalize=normalize)
    grayscale = color.rgb2gray(rgb_image)
    byte_image = img_as_ubyte(grayscale)

    angles, distances = get_rii_dist_angles()

    results_0 = greycomatrix(byte_image, angles[0], distances[0],
                             symmetric=False, normed=False, levels=256)
    results_1 = greycomatrix(byte_image, angles[1], distances[1],
                             symmetric=False, normed=False, levels=256)
    results_2 = greycomatrix(byte_image, angles[2], distances[2],
                             symmetric=False, normed=False, levels=256)

    return (results_0, results_1, results_2)
