import pytest

from satsense.image import Image


@pytest.fixture
def image():
    """
    Load image as a satsense image
    """
    # Sentinel is not actually quickbird, but I stored the bands this way
    return Image('test/data/source/section_2_sentinel.tif', 'quickbird')
