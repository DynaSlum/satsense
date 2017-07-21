"""
Methods for loading and saving mask images
"""
import rasterio
from scipy.misc import imread

def save_mask2file(mask, fullfname):
    w,h = mask.shape
    with rasterio.open(
        fullfname, 'w',
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