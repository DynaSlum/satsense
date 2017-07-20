import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty

from .feature import Feature
from ..bands import RGB, QUICKBIRD
from ..generators import CellGenerator


def rbNDVI(image, bands=RGB):
    """
    Calculates the red-blue normalized difference vegetation index of the image

    bands are assumed to be Red Green Blue, if different supply a bands array with the ordering
    """
    red_band = bands['red']
    blue_band = bands['blue']
    red = image[:, :, red_band].astype(np.float64)
    blue = image[:, :, blue_band].astype(np.float64)

    # Ignore divide, this division may complain about division by 0
    # This usually happens in the edge, which is alright by us.
    old_settings = np.seterr(divide='ignore', invalid='ignore')
    ndvi = np.divide(red - blue, red + blue)
    np.seterr(**old_settings)

    return ndvi


def rgNDVI(image, bands=RGB):
    """
    Calculates the red-green normalized difference vegetation index of the image

    bands are assumed to be Red Green Blue, if different supply a bands array with the ordering
    """
    red_band = bands['red']
    green_band = bands['green']
    red = image[:, :, red_band].astype(np.float64)
    green = image[:, :, green_band].astype(np.float64)

    # Ignore divide, this division may complain about division by 0
    # This usually happens in the edge, which is alright by us.
    old_settings = np.seterr(divide='ignore', invalid='ignore')
    ndvi = np.divide(red - green, red + green)
    np.seterr(**old_settings)

    return ndvi


def nirNDVI(image, bands=QUICKBIRD):
    """
    Calculates the red-green normalized difference vegetation index of the image

    bands are assumed to be Red Green Blue NIR, if different supply a bands array with the ordering
    """
    red_band = bands['red']
    infrared_band = bands['nir-1']
    red = image[:, :, red_band].astype(np.float64)
    nir = image[:, :, infrared_band].astype(np.float64)

    # Ignore divide, this division may complain about division by 0
    # This usually happens in the edge, which is alright by us.
    old_settings = np.seterr(divide='ignore', invalid='ignore')
    ndvi = np.divide(nir - red, nir + red)
    np.seterr(**old_settings)

    return ndvi


def print_ndvi_statistics(ndvi):
    """
    Prints the ndvi matrix and the, min, max, mean and median
    """
    print('NDVI matrix: ')
    print(ndvi)

    print('\nMax NDVI: {m}'.format(m=np.nanmax(ndvi)))
    print('Mean NDVI: {m}'.format(m=np.nanmean(ndvi)))
    print('Median NDVI: {m}'.format(m=np.nanmedian(ndvi)))
    print('Min NDVI: {m}'.format(m=np.nanmin(ndvi)))


class NDVI(Feature):
    __metaclass__ = ABCMeta

    def __init__(self, windows=((25, 25))):
        super(NDVI, self)
        self.windows = windows
        self.feature_size = len(self.windows)

    def __call__(self, cell):
        result = np.zeros(self.feature_size)
        for i, window in enumerate(self.windows):
            win = cell.super_cell(window, padding=True)
            ndvi = self.function(win.normalized, bands=cell.bands)
            result[i] = ndvi.mean()
        return result

    @abstractproperty
    def function(self):
        pass


class NirNDVI(NDVI):
    def __init__(self, windows=((25, 25))):
        super(NirNDVI, self).__init__(windows=windows)

    @property
    def function(self):
        return nirNDVI


class RgNDVI(NDVI):
    def __init__(self, windows=(25,)):
        super(RgNDVI, self).__init__(windows=windows)
    
    @property
    def function(self):
        return rgNDVI

class RbNDVI(NDVI):
    def __init__(self, windows=(25,)):
        super(RbNDVI, self).__init__(windows=windows)
    
    @property
    def function(self):
        return rbNDVI
