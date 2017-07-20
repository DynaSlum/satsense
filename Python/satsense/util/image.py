"""
Methods for loading images
"""
import rasterio

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
