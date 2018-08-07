"""
Methods for loading and saving mask images
"""
import rasterio
from scipy.misc import imread
from skimage.filters import threshold_otsu

from ..extract import extract_features_parallel
from ..features import FeatureSet, NirNDVI


def save_mask2file(mask, fullfname):
    w, h = mask.shape
    with rasterio.open(
            fullfname,
            'w',
            driver='GTiff',
            dtype=rasterio.uint8,
            count=1,
            width=w,
            height=h) as dst:
        dst.write(mask, indexes=1)


def load_mask_from_file(fullfname):
    """
    Loads a binary mask file from the specified full filename into a numpy array

    @returns mask The mask image loaded as a numpy array
    """
    mask = imread(fullfname)

    return mask


def get_ndxi_mask(generator, feature=NirNDVI):
    """Compute a mask based on an NDXI feature"""
    features = FeatureSet()
    windows = ((generator.x_size, generator.y_size), )
    features.add(feature(windows=windows))

    values = extract_features_parallel(features, generator)
    values.shape = (values.shape[0], values.shape[1])

    return values > threshold_otsu(values)
