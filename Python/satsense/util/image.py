"""
Methods for loading images
"""
from six import iteritems

import numpy as np
from osgeo import gdal
from osgeo import gdal_array

gdal.AllRegister()

def load_from_file(path):
    """
    Loads the specified path from file and loads the bands into a numpy array

    @returns dataset The raw gdal dataset
             image The image loaded as a numpy array
    """
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    image = np.rollaxis(dataset.ReadAsArray(), 0, 3).astype('float64')

    return dataset, image

def normalize_image(image, bands):
    """
    Normalizes the image based on the band maximum
    """
    normalized_image = image.copy()
    for name, band in iteritems(bands):
        # print("Normalizing band: {0}".format(name))
        mean = normalized_image[:, :, band].mean()
        std = normalized_image[:, :, band].std()

        new_min = mean - (2 * std)
        new_max = mean + (2 * std)

        normalized_image[normalized_image[:, :, band] > new_max, band] = new_max
        normalized_image[normalized_image[:, :, band] < new_min, band] = new_min

        normalized_image[:, :, band] = remap(normalized_image[:, :, band], new_min, new_max, 0, 1)

    return normalized_image

def get_rgb_image(image, bands, normalize=True):
    """
    Converts the image to rgb format.
    The image is normalized between 0 and 1.
    """
    if normalize:
        normalized_image = normalize_image(image, bands)
    else:
        normalized_image = image

    red = normalized_image[:, :, bands['red']]
    green = normalized_image[:, :, bands['green']]
    blue = normalized_image[:, :, bands['blue']]

    img = np.rollaxis(np.array([red, green, blue]), 0, 3)

    return img


def remap(x, o_min, o_max, n_min, n_max):
    # range check
    if o_min == o_max:
        print("Warning: Zero input range")
        return None

    if n_min == n_max:
        print("Warning: Zero output range")
        return None

    # check reversed input range
    reverse_input = False
    old_min = min(o_min, o_max)
    old_max = max(o_min, o_max)
    if not old_min == o_min:
        reverse_input = True

    #check reversed output range
    reverse_output = False
    new_min = min(n_min, n_max)
    new_max = max(n_min, n_max)
    if not new_min == n_min:
        reverse_output = True

    # print("Remapping from range [{0}-{1}] to [{2}-{3}]".format(old_min, old_max, new_min, new_max))
    portion = (x - old_min) * (new_max - new_min) / (old_max - old_min)
    if reverse_input:
        portion = (old_max - x) * (new_max - new_min) / (old_max - old_min)

    result = portion + new_min
    if reverse_output:
        result = new_max - portion

    return result
