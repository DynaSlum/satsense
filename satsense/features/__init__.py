from .feature import Feature, FeatureSet
from .hog import HistogramOfGradients, ScipyHoG
from .lacunarity import Lacunarity
from .ndxi import NDSI, NDWI, NDXI, WVSI, NirNDVI, RbNDVI, RgNDVI
from .pantex import Pantex
from .sift import Sift
from .texton import Texton

# Change the module for base classes so sphinx can find them.
Feature.__module__ = __name__
NDXI.__module__ = __name__

__all__ = [
    'Feature',
    'FeatureSet',
    'HistogramOfGradients',
    'ScipyHoG',
    'Pantex',
    'NDXI',
    'NirNDVI',
    'RgNDVI',
    'RbNDVI',
    'NDSI',
    'NDWI',
    'WVSI',
    'Lacunarity',
    'Sift',
    'Texton',
]
