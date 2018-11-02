import rasterio
import pytest
import numpy as np


@pytest.mark.skip(reason="Not a  test")
def normalized_image():
    """
    Get the test image and normalize it
    """
    with rasterio.open('../data/section_2_sentinel.tif') as dataset:
        image_in = dataset.read(masked=True).astype('float32')
        image = np.empty_like(image_in)

        # Normalization
        percentiles = [2, 98]
        for i in range(dataset.count):
            band = image_in[i]
            data = band[~band.mask]

            lower, upper = np.nanpercentile(data, percentiles)
            band -= lower
            band /= upper - lower
            np.clip(band, a_min=0, a_max=1, out=band)

            image[i] = band

        return image, dataset.crs, dataset.transform


@pytest.mark.skip(reason="Not a  test")
def write_target(target_image, image_name, crs, transform):
    with rasterio.open(image_name, 'w', driver='GTiff',
                       height=target_image.shape[0], width=target_image.shape[1],
                       count=1, dtype=str(target_image.dtype), crs=crs, transform=transform,
                       nodata=target_image.fill_value) as target:
        target.write(target_image, 1)
        target.write_mask(~target_image.mask)


@pytest.mark.skip(reason="Not a  test")
def ndxi_target(b1, b2, name):
    image, crs, transform = normalized_image()

    band1 = image[b1]
    band2 = image[b2]

    target = np.divide(band1 - band2, band1 + band2)

    write_target(target, 'target_' + name + '.tif', crs, transform)


@pytest.mark.skip(reason="Not a  test")
def ndvi_target():
    ndxi_target(3, 2, 'ndvi')


@pytest.mark.skip(reason="Not a  test")
def rg_ndvi_target():
    ndxi_target(2, 1, 'rg_ndvi')


@pytest.mark.skip(reason="Not a  test")
def rb_ndvi_target():
    ndxi_target(2, 0, 'rb_ndvi')


@pytest.mark.skip(reason="Not a  test")
def ndsi_target():
    ndxi_target(3, 1, 'ndsi')


ndvi_target()
rg_ndvi_target()
rb_ndvi_target()
ndsi_target()
