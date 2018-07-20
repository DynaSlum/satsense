import fiona
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio import mask
from scipy.ndimage import zoom
from skimage import filters

from satsense.bands import MASK_BANDS
from satsense.generators import CellGenerator
from satsense.image import SatelliteImage

from ..features import WVSI, NirNDVI
from ..util import load_mask_from_file, save_mask2file


class Mask():
    def __init__(self, mask):
        self._mask = np.array(mask)

    @property
    def shape(self):
        return self._mask.shape

    @property
    def mask(self):
        return self._mask
    
    def save(self, path):
        np.save(path, self._mask)

    def load_from_file_tif(path):
        mask = load_mask_from_file(path)
        return Mask(mask)

    def load_from_file(path):
        mask = np.load(path)
        return Mask(mask)

    def overlay(self, rgb_image):
        zoom_w = rgb_image.shape[1] / self.mask.shape[1]
        zoom_h = rgb_image.shape[0] / self.mask.shape[0]
        zoomed_mask = zoom(self.mask, (zoom_h, zoom_w), order=0)

        plt.imshow(rgb_image)
        plt.imshow(zoomed_mask, cmap='hot', alpha=0.3)

    def __and__(self, other):
        return Mask(np.uint8(np.logical_and(self.mask, other.mask)))

    def __or__(self, other):
        return Mask(np.uint8(np.logical_or(self.mask, other.mask)))

    def __invert__(self):
        return Mask(np.uint8(np.logical_not(self.mask)))

    def __sub__(self, other):
        """ A & ~B
            Removes everything from A that is present in B. This is used to
            remove slums from the vegetation mask.
        """
        m = np.logical_and(self.mask, np.logical_not(other.mask))
        return Mask(np.uint8(m))

    def resample(self, size, threshold=0.8):
        # Need 3d instead of 2d for cell generator
        tmp = self.mask[:, :, np.newaxis]
        mask_image = SatelliteImage(None, tmp, MASK_BANDS)
        
        generator = CellGenerator(mask_image, size)

        resampled_mask = np.zeros(generator.shape)
        for cell in generator:
            if np.mean(cell.raw) > threshold:
                    resampled_mask[cell.x, cell.y] = 1
        return resampled_mask 

class VegetationMask(Mask):   
    @staticmethod 
    def create(generator):
        mask = np.zeros(generator.shape)
        ndvi = NirNDVI(windows=((1, 1), ))
        for cell in generator:
            mask[cell.x, cell.y] = ndvi(cell)
        mask = np.uint8(mask < filters.threshold_otsu(mask))
        return VegetationMask(mask)


class SoilMask(Mask):
    @staticmethod 
    def create(generator):
        mask = np.zeros(generator.shape)
        wvsi = WVSI(windows=((1, 1), ))
        for cell in generator:
            mask[cell.x, cell.y] = wvsi(cell)
        mask = np.uint8(mask > filters.threshold_otsu(mask))
        return VegetationMask(mask)

class OnesMask(Mask):
    @staticmethod
    def create(generator):
        mask = np.ones(generator.shape, dtype=np.uint8)
        return OnesMask(mask)

class ShapefileMask(Mask):
    def create(shapefile, imagefile, size=(1,1)):
        with fiona.open(shapefile, "r") as sf:
            geoms = [feature["geometry"] for feature in sf]

        with rasterio.open(imagefile) as src:
            out_image, _ = rasterio.mask.mask(src, geoms, crop=False,
                                              invert=False)
            
            # Set the 'no-data' value to 0 instead of 3.4e+38
            out_image[out_image == np.max(out_image)] = 0
            # Set all values in the masked area to one as the masked area
            # retains the original values 
            out_image[out_image > 0] = 1
            # The same mask is created for every band, therefore the first
            # band is selected 
            out_image = out_image[0]

        return ShapefileMask(np.uint8(out_image))
