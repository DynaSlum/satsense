"""
Methods for loading images
"""
from six import iteritems

import numpy as np
from osgeo import gdal

from skimage import color, img_as_ubyte
from .bands import RGB

import warnings

gdal.AllRegister()


def load_from_file(path):
    """
    Loads the specified path from file and loads the bands into a numpy array

    @returns dataset The raw gdal dataset
             image The image loaded as a numpy array
    """
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    array = dataset.ReadAsArray()
    if len(array.shape) == 3:
        # The bands column is in the first position, but we want it last
        array = np.rollaxis(array, 0, 3)
    elif len(array.shape) == 2:
        # This image seems to have one band, so we add an axis for ease
        # of use in the rest of the library
        array = array[:, :, np.newaxis]

    image = array.astype('float64')

    return dataset, image


def normalize_image(image, bands, technique='cumulative',
                    percentiles=[2.0, 98.0], numstds=2):
    """
    Normalizes the image based on the band maximum
    """
    normalized_image = image.copy()
    for name, band in iteritems(bands):
        # print("Normalizing band number: {0} {1}".format(band, name))
        if technique == 'cumulative':
            percents = np.percentile(image[:, :, band], percentiles)
            new_min = percents[0]
            new_max = percents[1]
        elif technique == 'meanstd':
            mean = normalized_image[:, :, band].mean()
            std = normalized_image[:, :, band].std()

            new_min = mean - (numstds * std)
            new_max = mean + (numstds * std)
        else:
            new_min = normalized_image[:, :, band].min()
            new_max = normalized_image[:, :, band].max()

        if new_min:
            normalized_image[normalized_image[:, :, band] < new_min, band] = new_min
        if new_max:
            normalized_image[normalized_image[:, :, band] > new_max, band] = new_max

        normalized_image[:, :, band] = remap(normalized_image[:, :, band], new_min, new_max, 0, 1)

    return normalized_image


def get_rgb_bands(image, bands):
    """
    Converts the image to rgb format.
    """
    red = image[:, :, bands['red']]
    green = image[:, :, bands['green']]
    blue = image[:, :, bands['blue']]

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

    # check reversed output range
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


def get_grayscale_image(image, bands=RGB):
    if bands is not RGB:
        rgb_image = get_rgb_bands(image, bands)
    else:
        rgb_image = image

    return color.rgb2gray(rgb_image)


def get_ubyte_image(image, bands=RGB):
    """
    Converts image in 0 - 1 scale format to ubyte 0 - 255 format

    Uses img_as_ubyte from skimage
    """
    if bands is not RGB:
        rgb_image = get_rgb_bands(image, bands)
    else:
        rgb_image = image

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore loss of precision warning
        return img_as_ubyte(rgb_image)