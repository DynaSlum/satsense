"""Methods for loading images."""

import logging
import math
import time
import warnings
from types import MappingProxyType

import numpy as np
import rasterio
from affine import Affine
from netCDF4 import Dataset
from pyproj import Proj, transform
from skimage import img_as_ubyte
from skimage.color import gray2rgb, rgb2gray

from . import __version__
from .bands import BANDS

logger = logging.getLogger(__name__)


class Image:

    itypes = {}

    @classmethod
    def register(cls, itype, function):
        """Register a new image type."""
        cls.itypes[itype] = function

    def __init__(self,
                 filename,
                 satellite,
                 band='rgb',
                 normalization_parameters=None,
                 block=None,
                 cached=None):

        self.filename = filename
        self.satellite = satellite
        self.bands = BANDS[satellite.lower()]
        self.band = band

        self.normalization = {}
        if normalization_parameters is None:
            normalization_parameters = {
                'technique': 'cumulative',
                'percentiles': [2.0, 98.0],
                'dtype': np.float32,
            }
        self.normalization_parameters = normalization_parameters

        self._block = block
        self.cached = [] if cached is None else cached
        self.cache = {}

        self.attributes = {}

    def copy_block(self, block):
        """Create a subset of Image."""
        logger.info("Selecting block %s from image with shape %s", block,
                    self.shape)
        image = Image(
            self.filename,
            self.satellite,
            band=self.band,
            normalization_parameters=self.normalization_parameters,
            block=block,
            cached=self.cached)
        image.normalization = MappingProxyType(self.normalization)
        return image

    def __getitem__(self, itype):
        """Get image of type."""
        if itype in self.cache:
            return self.cache[itype]

        if itype in self.bands:
            image = self._load_band(itype, self._block)
        elif itype in self.itypes:
            image = self.itypes[itype](self)
        else:
            raise IndexError(
                "Unknown itype {}, choose from {} or register a new itype "
                "using the register method.".format(itype, self.itypes))

        if self.cached is True or itype in self.cached:
            self.cache[itype] = image

        return image

    def _load_band(self, band, block=None):
        """Read band from file and normalize if required."""
        image = self._read_band(band, block)
        if self.normalization_parameters:
            dtype = self.normalization_parameters['dtype']
            image = image.astype(dtype, casting='same_kind', copy=False)
            self._normalize(image, band)
        return image

    def _read_band(self, band, block=None):
        """Read band from file."""
        logger.info("Loading band %s from file %s", band, self.filename)
        bandno = self.bands[band] + 1
        with rasterio.open(self.filename) as dataset:
            image = dataset.read(
                bandno, window=block, boundless=True, masked=True)
            return image

    def precompute_normalization(self, *bands):

        if not self.normalization_parameters:
            return

        for band in bands or self.bands:
            if band not in self.normalization:
                self._get_normalization_limits(band)

    def _get_normalization_limits(self, band, image=None):
        """Return normalization limits for band."""
        if band not in self.normalization:
            # select only non-masked values for computing scale
            if image is None:
                overwrite_input = True
                image = self._read_band(band)
            else:
                overwrite_input = False
            data = image[~image.mask] if np.ma.is_masked(image) else image
            technique = self.normalization_parameters['technique']
            if not data.any():
                limits = 0, 0
            elif technique == 'cumulative':
                percentiles = self.normalization_parameters['percentiles']
                limits = np.nanpercentile(
                    data, percentiles, overwrite_input=overwrite_input)
            elif technique == 'meanstd':
                numstds = self.normalization_parameters['numstds']
                mean = data.nanmean()
                std = data.nanstd()
                limits = mean - (numstds * std), mean + (numstds * std)
            else:
                limits = data.nanmin(), data.nanmax()

            lower, upper = limits
            logger.info("Normalizing [%s, %s] to [0, 1] for band %s", lower,
                        upper, band)
            if not np.isclose(lower, upper) and lower > upper:
                raise ValueError(
                    "Unable to normalize {} band of {} with normalization "
                    "parameters {} because lower limit is larger or equal to "
                    "upper limit for limits={}.".format(
                        band, self.filename, self.normalization_parameters,
                        limits))

            self.normalization[band] = limits

        return self.normalization[band]

    def _normalize(self, image, band):
        """Normalize image with limits for band."""
        lower, upper = self._get_normalization_limits(band, image)
        if np.isclose(lower, upper):
            logger.warning(
                "Lower and upper limit %s, %s are considered too close "
                "to normalize band %s, setting it to 0.", lower, upper, band)
            image[:] = 0
        else:
            image -= lower
            image /= upper - lower
            np.ma.clip(image, a_min=0, a_max=1, out=image)

    def _get_attribute(self, key):
        if key not in self.attributes:
            with rasterio.open(self.filename) as dataset:
                self.attributes[key] = getattr(dataset, key)
        return self.attributes[key]

    @property
    def shape(self):
        return self._get_attribute('shape')

    @property
    def crs(self):
        return self._get_attribute('crs')

    @property
    def transform(self):
        return self._get_attribute('transform')

    def scaled_transform(self, step_size):
        return self.transform * Affine.scale(*step_size) * Affine.translation(
            0.5, 0.5)


def get_rgb_image(image: Image):
    """Convert the image to rgb format."""
    #     logger.debug("Computing rgb image")
    if image.band == 'rgb':
        red = image['red']
        green = image['green']
        blue = image['blue']
        rgb = np.ma.dstack([red, green, blue])
    else:
        gray = image[image.band]
        rgb = np.ma.array(gray2rgb(gray))
        rgb.mask = gray.mask
    #     logger.debug("Done computing rgb image")
    return rgb


Image.register('rgb', get_rgb_image)


def get_grayscale_image(image: Image):
    #     logger.debug("Computing grayscale image")
    if image.band == 'rgb':
        rgb = image['rgb']
        mask = np.bitwise_or.reduce(rgb.mask, axis=2)
        gray = np.ma.array(rgb2gray(rgb), mask=mask)
    else:
        gray = image[image.band]
    #     logger.debug("Done computing grayscale image")
    return gray


Image.register('grayscale', get_grayscale_image)


def get_gray_ubyte_image(image: Image):
    """Convert image in 0 - 1 scale format to ubyte 0 - 255 format.

    Uses img_as_ubyte from skimage.
    """
    #     logger.debug("Computing gray ubyte image")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Ignore loss of precision warning
        gray = image['grayscale']
        ubyte = np.ma.array(img_as_ubyte(gray), mask=gray.mask)

    #     logger.debug("Done computing gray ubyte image")
    return ubyte


Image.register('gray_ubyte', get_gray_ubyte_image)


class FeatureVector():
    """Class to store a feature vector in."""

    def __init__(self, feature, vector, crs=None, transform=None):
        self.vector = vector
        self.feature = feature

        self.crs = crs
        self.transform = transform

    def get_filename(self, window, prefix='', extension='nc'):
        return '{}{}_{}_{}.{}'.format(prefix, self.feature.name, window[0],
                                      window[1], extension)

    def save(self, filename_prefix='', extension='nc'):
        """Save feature vector to file."""
        filenames = []
        for i, window in enumerate(self.feature.windows):
            filename = self.get_filename(window, filename_prefix, extension)
            filenames.append(filename)
            logger.info("Saving feature %s window %s to file %s",
                        self.feature.name, window, filename)

            data = self.vector[..., i, :]
            if extension.lower() == 'nc':
                self._save_as_netcdf(data, filename, window)
            elif extension.lower() == 'tif':
                self._save_as_tif(data, filename, window)
        return filenames

    def _save_as_netcdf(self, data, filename, window):
        """Save feature vector as NetCDF file."""
        if len(data.shape) == 2:
            width, height = data.shape
            value_length = 1
            data = data[:, :, np.newaxis]
        elif len(data.shape) == 3:
            width, height, value_length = data.shape

        with Dataset(filename, 'w') as dataset:
            # Metadata
            dataset.history = 'Created ' + time.ctime(time.time())
            dataset.source = 'Satsense version ' + __version__
            dataset.description = (
                'Satsense extracted values for feature: ' + self.feature.name)
            dataset.title = self.feature.name
            dataset.window = window
            dataset.arguments = repr(self.feature.kwargs)

            # Latitude and Longitude variables
            dataset.createDimension('lon', height)
            dataset.createDimension('lat', width)

            lats = dataset.createVariable('lat', 'f8', dimensions=('lat'))
            lats.standard_name = 'latitude'
            lats.long_name = 'latitude'
            lats.units = 'degree_north'
            lats._CoordinateAxisType = "Lat"  # noqa W0212
            lons = dataset.createVariable('lon', 'f8', dimensions=('lon'))
            lons.standard_name = 'longitude'
            lons.long_name = "longitude"
            lons.units = 'degrees_east'
            lons._CoordinateAxisType = "Lon"  # noqa W0212

            crs = dataset.createVariable('spatial_ref', 'i4')
            crs.spatial_ref = """GEOGCS["WGS 84",
                                        DATUM["WGS_1984",
                                              SPHEROID["WGS 84",
                                              6378137,
                                              298.257223563,
                                              AUTHORITY["EPSG","7030"]
                                        ],
                                        AUTHORITY["EPSG","6326"]
                                    ],
                                    PRIMEM["Greenwich", 0,
                                            AUTHORITY["EPSG","8901"]
                                    ],
                                    UNIT["degree",0.0174532925199433,
                                            AUTHORITY["EPSG","9122"]
                                    ],
                                    AUTHORITY["EPSG","4326"]
                              ]"""

            # Transform the cell indices to lat/lon based on the image crs
            # and transform Based on
            # https://gis.stackexchange.com/questions/129847/obtain-coordinates-and-corresponding-pixel-values-from-geotiff-using-python-gdal
            t1 = self.transform
            cols, rows = np.arange(height), np.arange(width)

            rc2en = lambda r, c: (c, r) * t1
            eastings, northings = np.vectorize(
                rc2en, otypes=[np.float, np.float])(rows, cols)

            p1 = Proj(self.crs)
            p2 = Proj(proj='latlong', datum='WGS84')
            long_values, lat_values = transform(p1, p2, eastings, northings)

            lons[:] = long_values
            lats[:] = lat_values

            # Actually add the values
            dataset.createDimension('length', value_length)
            variable = dataset.createVariable(
                self.feature.name, 'f4', dimensions=('length', 'lat', 'lon'))
            variable.grid_mapping = 'spatial_ref'
            variable.long_name = self.feature.name

            transposed = np.transpose(data, (2, 0, 1))
            variable[:] = transposed

    def _save_as_tif(self, data, filename, window):
        """Save feature array as GeoTIFF file."""
        width, height, size = data.shape
        data = np.ma.filled(data)
        fill_value = data.fill_value if np.ma.is_masked(data) else None
        # This is probably wrong
        data = np.moveaxis(data, source=2, destination=0)
        with rasterio.open(
                filename,
                mode='w',
                driver='GTiff',
                width=width,
                height=height,
                count=size,
                dtype=rasterio.float32,
                crs=self.crs,
                transform=self.transform,
                nodata=fill_value,
        ) as dataset:
            dataset.write(data)
            dataset.update_tags(
                window=window, arguments=repr(self.feature.kwargs))

    @classmethod
    def from_file(cls, feature, filename_prefix, extension='nc'):
        """Restore saved features."""
        new = cls(feature, None)
        for window in feature.windows:
            filename = new.get_filename(window, filename_prefix, extension)
            logger.debug("Loading feature %s from file %s", feature.name,
                         filename)
            with Dataset(filename, 'r') as dataset:
                var = dataset.variables[feature.name]
                if new.vector is None:
                    shape = var.shape[:2] + (len(feature.windows),
                                             feature.size)
                    new.vector = np.zeros(shape, dtype=np.float32)
                window = tuple(dataset.window)
                idx = feature.windows.index(window)
                if repr(feature.kwargs) != dataset.arguments:
                    logger.warning(
                        "Stored arguments do not match feature, %r != %s",
                        feature.kwargs, dataset.arguments)
                new.vector[:, :, idx, :] = var[:]
        return new
