"""
Methods for loading images
"""
import numpy as np
from osgeo import gdal

def load_from_file(path):
    """
    Loads the specified path from file and loads the bands into a numpy array

    @returns dataset The raw gdal dataset
             image The image loaded as a numpy array
    """
    dataset = gdal.Open(path, gdal.GA_ReadOnly)

    band = dataset.GetRasterBand(1)

    # print(dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount)
    image = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount),
                     dtype=np.float64)

    # Loop over all bands in dataset
    for b in range(dataset.RasterCount):
        # Remember, GDAL index is on 1, but Python is on 0 -- so we add 1 for our GDAL calls
        band = dataset.GetRasterBand(b + 1)
        # print("Band: {0} {1}".format(b, band))

        # Read in the band's data into the third dimension of our array
        image[:, :, b] = band.ReadAsArray()

        # print('band {0} min: {1}, max: {2}'.format(b, image[:, :, b].min(), image[:, :, b].max()))

    return dataset, image

def normalize_image(image, axis=2):
    normalized_image = image.copy()
    for b in range(image.shape[axis]):
        maximum = normalized_image[:, :, b].max()
        normalized_image[:, :, b] /= maximum
    return normalized_image

def get_rgb_image(image, bands, normalized=False):
    if not normalized:
        normalized_image = normalize_image(image)
    else:
        normalized_image = image
    red = normalized_image[:, :, bands['red']]
    green = normalized_image[:, :, bands['green']]
    blue = normalized_image[:, :, bands['blue']]

    img = np.rollaxis(np.array([red, green, blue]), 0, 3)

    return img
