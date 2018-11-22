"""
Testing features
"""
import rasterio
import pytest
from satsense.features.ndxi import ndxi_image
from satsense.image import Image


@pytest.fixture
def image():
    # Sentinel is not actually quickbird, but I stored the bands this way
    image = Image('test/data/section_2_sentinel.tif', 'quickbird')

    return image


def test_ndvi(image):
    """
    Test ndvi
    """
    with rasterio.open('test/data/target_ndvi.tif') as dataset:
        target = dataset.read(masked=True)

        result = ndxi_image(image, 'nir_ndvi')

        same = result == target

        assert same.all()


def test_rg_ndvi(image):
    """
    Test ndvi
    """
    with rasterio.open('test/data/target_rg_ndvi.tif') as dataset:
        target = dataset.read(masked=True)

        result = ndxi_image(image, 'rg_ndvi')

        same = result == target

        assert same.all()


def test_rb_ndvi(image):
    """
    Test ndvi
    """
    with rasterio.open('test/data/target_rb_ndvi.tif') as dataset:
        target = dataset.read(masked=True)

        result = ndxi_image(image, 'rb_ndvi')

        same = result == target

        assert same.all()


def test_ndsi(image):
    """
    Test ndvi
    """
    with rasterio.open('test/data/target_ndsi.tif') as dataset:
        target = dataset.read(masked=True)

        result = ndxi_image(image, 'ndsi')

        same = result == target

        assert same.all()
