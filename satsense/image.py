"""Methods for loading images."""

import logging
import os
import time
import warnings
from ast import literal_eval as make_tuple
from pathlib import Path
from types import MappingProxyType

import numpy as np
import rasterio
from affine import Affine
from netCDF4 import Dataset
from skimage import img_as_ubyte
from skimage.color import gray2rgb, rgb2gray

from . import __version__
from .bands import BANDS

logger = logging.getLogger(__name__)


class Image:
    """
    Image class that provides a unified interface to satellite images.

    Under the hood rasterio is used, so any format supported by rasterio
    can be used.

    Parameters
    ----------
    filename: str
        The name of the image
    satellite: str
        The name of the satelite (i.e. worldview3, quickbird etc.)
    band: str
        The band for the grayscale image, or 'rgb'. The default is 'rgb'
    normalization_parameters: dict or boolean, optional
        if False no normalization is done.
        if None the default normalization will be applied
        (cumulative with 2, 98 percentiles)

        f a Dictionary that describes the normalization parameters
        The following keys can be supplied:

        - technique: string
            The technique to use, can be 'cumulative' (default),
            'meanstd' or 'minmax'
        - percentiles: list[int]
            The percentiles to use (exactly 2) if technique is cumulative,
            default is [2, 98]
        - numstds: float
            Number of standard deviations to use if technique is meanstd
    block: tuple or rasterio.windows.Window, optional
        The part of the image read defined in a rasterio compatible way,
        e.g. two tuples or a rasterio.windows.Window object
    cached: array-like or boolean, optional
        If True bands and base images are cached in memory
        if an array a band or base image is cached if its name is in the
        array

    Examples
    ========
    Load an image and inspect the shape and bands

    from satsense import Image
    >>> image = Image('test/data/source/section_2_sentinel.tif', 'quickbird')
    >>> image.shape
    (152, 155)

    >>> image.bands
    {'blue': 0, 'green': 1, 'red': 2, 'nir-1': 3}

    >>> image.crs
    CRS({'init': 'epsg:32643'})

    See also
    ========
    satsense.bands
    """

    itypes = {}

    @classmethod
    def register(cls, itype, function):
        """
        Register a new image type.

        Parameters
        ----------
        itype: str
            (internal) name of the image type
        function
            Function definition that should take a single Image parameter
            and return a numpy.ndarray or numpy.ma.masked_array

        See Also
        --------
        :ufunc: get_gray_ubyte_image
        :ufunc: get_grayscale_image
        :ufunc: get_rgb_image
        """
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
        """
        Create a subset of Image.

        Parameters
        ----------
        block: tuple or rasterio.windows.Window
            The part of the image to read defined in a rasterio compatible way,
            e.g. two tuples or a rasterio.windows.Window object

        Returns
        -------
        image.Image:
            subsetted image
        """
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
        """
        Get image of a type registered using the `register` method.
        The following itypes are available to facilitate creating
        new features:
        - 'rgb'
        - 'grayscale'
        - 'gray_ubyte'

        Parameters
        ----------
        itype: str
            The name of the image type to retrieve

        Returns
        -------
        out: numpy.ndarray or numpy.ma.masked_array
            The image of the supplied type

        Examples
        --------
        Get the rgb image
        >>> image['rgb'].shape
        (152, 155, 3)
        >>> image['gray_ubyte'].dtype
        dtype('uint8')
        """
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
        """
        Read band from file and normalize if required.

        Parameters
        ----------
        band: str
            The band of the grayscale image, or 'rgb'
        block: tuple or rasterio.windows.Window, optional
            The part of the image read defined in a rasterio compatible way

        Returns
        -------
            The loaded and normalized band
        """
        image = self._read_band(band, block)
        if self.normalization_parameters:
            dtype = self.normalization_parameters['dtype']
            image = image.astype(dtype, casting='same_kind', copy=False)
            self._normalize(image, band)
        return image

    def _read_band(self, band, block=None):
        """
        Read spectral band from file.

        Parameters
        ----------
        band: str
            The band of the grayscale image, or 'rgb'
        block: tuple or rasterio.windows.Window, optional
            The part of the image read defined in a rasterio compatible way

        Returns
        -------
            The loaded band with the extent as supplied by block
        """
        logger.info("Loading band %s from file %s", band, self.filename)
        bandno = self.bands[band] + 1
        with rasterio.open(self.filename) as dataset:
            image = dataset.read(
                bandno, window=block, boundless=True, masked=True)
            return image

    def precompute_normalization(self, *bands):
        """
        Precompute the normalization of the image

        Normalization is done using the normalization_parameters supplied
        during class instantiation. Normalization parameters are computed
        automatically for all bands when required, but doing it explicitly
        can save some time, e.g. if there are more bands in the image than
        needed.

        Parameters
        ==========
        *bands : list[str] or None
            The list of bands to normalize, if None all bands will be
            normalized

        Raises
        ======
        ValueError:
            When trying to compute the normalization on a partial image,
            as created by using the `copy_block` method.

        See Also
        ========
        Image
            :func: _normalize
            Get normalization limits for band(s).
        """
        if not self.normalization_parameters:
            return

        for band in bands or self.bands:
            if band not in self.normalization:
                self._get_normalization_limits(band)

    def _get_normalization_limits(self, band, image=None):
        """
        Return normalization limits for band.

        Parameters
        ----------
        band: str
            The band of the grayscale image, or 'rgb'
        image: numpy.ndarray or numpy.ma.masked_array, optional
            Image to normalize
        """
        if band not in self.normalization:
            if isinstance(self.normalization, MappingProxyType):
                raise ValueError(
                    "Unable to compute normalization on part of the image. "
                    "Please use the precompute_normalization() method of "
                    "the full image.")

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
        """
        Normalize image with limits for band.

        Parameters
        ----------
        image: numpy.ndarray or numpy.ma.masked_array
            Image to normalize
        band: str
            The band of the grayscale image, or 'rgb'
        """
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
        """Provide shape attribute."""
        return self._get_attribute('shape')

    @property
    def crs(self):
        """Provide crs attribute."""
        return self._get_attribute('crs')

    @property
    def transform(self):
        """Provide transform attribute."""
        return self._get_attribute('transform')

    def scaled_transform(self, step_size):
        """
        Perform a scaled transformation.

        Returns
        -------
            out : affine.Affine
                An affine transformation scaled by the step size
        """
        return self.transform * Affine.scale(*step_size)


def get_rgb_image(image: Image):
    """
    Convert the image to rgb format.

    Parameters
    ==========
    image : image.Image
        The image to calculate the rgb image from

    Returns
    -------
    numpy.ndarray
        The image converted to rgb
    """
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
    """
    Convert the image to grayscale.

    Parameters
    ==========
    image : image.Image
        The image to calculate the grayscale image from

    Returns
    -------
    numpy.ndarray
        The image converted to grayscale in 0 - 1 range

    See Also
    --------
    skimage.color.rgb2gray:
        Used to convert rgb image to grayscale
    """
    #     logger.debug("Computing grayscale image")
    if image.band == 'rgb':
        rgb = image['rgb']
        mask = np.bitwise_or.reduce(rgb.mask, axis=2)
        gray = np.ma.array(rgb2gray(rgb), mask=mask, dtype=np.float32)
    else:
        gray = image[image.band]
    #     logger.debug("Done computing grayscale image")
    return gray


Image.register('grayscale', get_grayscale_image)


def get_gray_ubyte_image(image: Image):
    """
    Convert image in 0 - 1 scale format to ubyte 0 - 255 format.

    Parameters
    ----------
    image: image.Image
        The image to calculate the grayscale image from

    Returns
    -------
    numpy.ndarray
        The image converted to grayscale

    See Also
    --------
    skimage.img_as_ubyte:
        Used to convert the image to ubyte
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
    """
    Class to store a feature vector in.

    Parameters
    ----------
    feature : satsense.feature.Feature
        The feature to store
    vector : array-like
        The data of the computed feature
    crs :
        The coordinate reference system for the data
    transform : Affine
        The affine transformation for the data
    """

    def __init__(self, feature, vector, crs=None, transform=None):
        self.feature = feature
        self.vector = vector
        self.crs = crs
        self.transform = transform

    def get_filename(self, window, prefix='', extension='nc'):
        """
        Construct filename from input parameters.

        Parameters
        ----------
        window: tuple
            The shape of the window used to calculate the feature
        prefix: str
            Prefix for the filename
        extension: str
            Filename extension

        Returns
        -------
            str
        """
        if os.path.isdir(prefix) and not str(prefix).endswith(os.sep):
            prefix += os.sep
        return '{}{}_{}_{}.{}'.format(prefix, self.feature.name, window[0],
                                      window[1], extension)

    def save(self, filename_prefix='', extension='nc'):
        """
        Save feature vectors to files.

        Parameters
        ----------
        filename_prefix: str
            Prefix for the filename
        extension: str
            Filename extension
        Returns:
            1-D array-like (str)
        """
        filenames = []
        for i, window in enumerate(self.feature.windows):
            filename = self.get_filename(window, filename_prefix, extension)
            filenames.append(filename)
            logger.info("Saving feature %s window %s to file %s",
                        self.feature.name, window, filename)

            data = self.vector[..., i, :]
            data = np.moveaxis(data, source=2, destination=0)

            description = 'Satsense extracted values for feature: {}'.format(
                self.feature.name)
            attributes = {
                'history': 'Created ' + time.ctime(),
                'source': 'Satsense version ' + __version__,
                'description': description,
                'title': self.feature.name,
                'window': window,
                'arguments': repr(self.feature.kwargs),
            }
            if extension.lower() == 'nc':
                self._save_as_netcdf(data, filename, attributes)
            elif extension.lower() == 'tif':
                self._save_as_tif(data, filename, attributes)
        return filenames

    def _save_as_netcdf(self, data, filename, attributes):
        """
        Save feature vector as NetCDF file.

        Parameters
        ----------
        data: np.array
            Feature vector
        filename: str
            Filename to save to
        attributes: dict of attributes
            Attributes to set in NetCDF file
        """
        feature_size, height, width = data.shape

        with Dataset(filename, 'w') as dataset:
            for attr in attributes:
                setattr(dataset, attr, attributes[attr])

            dimensions = self._add_lat_lon_dimensions(dataset, height, width)

            # Actually add the values
            dataset.createDimension('length', feature_size)
            variable = dataset.createVariable(
                self.feature.name, 'f4', dimensions=('length', *dimensions))
            variable.grid_mapping = 'spatial_ref'
            variable.long_name = self.feature.name

            variable[:] = data

    def _save_as_tif(self, data, filename, attributes):
        """
        Save feature array as GeoTIFF file.

        Parameters
        ----------
        data: np.array
            Feature vector
        filename: str
            Filename to save to
        attributes: dict of attributes
            Attributes to set in NetCDF file
        """
        feature_size, height, width = data.shape
        if np.ma.is_masked(data):
            fill_value = data.fill_value
            data = data.filled()
        else:
            fill_value = None

        with rasterio.open(
                filename,
                mode='w',
                driver='GTiff',
                width=width,
                height=height,
                count=feature_size,
                dtype=data.dtype,
                crs=self.crs,
                transform=self.transform,
                nodata=fill_value,
        ) as dataset:
            dataset.write(data)
            dataset.update_tags(**attributes)

    @classmethod
    def from_file(cls, feature, filename_prefix):
        """
        Restore saved features.

        Parameters
        ----------
        feature : Feature
            The feature to restore from a file
        filename_prefix : str
            The directory and other prefixes to find the feature file at

        Returns
        -------
        satsense.image.FeatureVector
            The feature loaded into a FeatureVector object
        """
        new = cls(feature, None)
        for window in feature.windows:
            for ext in ('nc', 'tif'):
                filename = new.get_filename(window, filename_prefix, ext)
                if Path(filename).is_file():
                    logger.debug("Loading feature %s from file %s",
                                 feature.name, filename)
                    break
            else:
                raise ValueError(
                    "Could not find a file containing feature {} in {}".format(
                        feature.name, filename_prefix))

            if Path(filename).suffix == '.nc':
                with Dataset(filename, 'r') as dataset:
                    data = np.ma.array(dataset.variables[feature.name][:])
                    window = tuple(dataset.window)
                    arguments = dataset.arguments
            else:
                with rasterio.open(filename, 'r') as dataset:
                    data = dataset.read(masked=True)
                    window = make_tuple(dataset.tags()['window'])
                    arguments = dataset.tags()['arguments']

            if repr(feature.kwargs) != arguments:
                logger.warning(
                    "Stored arguments do not match feature, %r != %s",
                    feature.kwargs, arguments)

            if new.vector is None:
                shape = data.shape[1:] + (len(feature.windows), feature.size)
                new.vector = np.ma.zeros(shape, dtype=data.dtype)
                new.vector.mask = np.zeros(shape, dtype=bool)

            idx = feature.windows.index(window)
            data = np.moveaxis(data, source=0, destination=2)
            new.vector[:, :, idx, :] = data
            new.vector.mask[:, :, idx, :] = data.mask

        return new

    def _add_lat_lon_dimensions(self, dataset, height, width):
        """
        Add longitude and latitude dimensions to dataset.

        Parameters
        ----------
        dataset: netCDF4._netCDF4.Dataset
            netCDF4 dataset
        height: int
            height of dataset
        width: int
            width of dataset
        """
        if self.crs.is_geographic:
            # Latitude and Longitude variables
            dataset.createDimension('lon', width)
            dataset.createDimension('lat', height)

            lats = dataset.createVariable('lat', 'f8', dimensions=('lat'))
            lons = dataset.createVariable('lon', 'f8', dimensions=('lon'))

            lats.standard_name = 'latitude'
            lats.long_name = 'latitude'
            lats.units = 'degrees_north'
            lats._CoordinateAxisType = "Lat"  # noqa W0212

            lons.standard_name = 'longitude'
            lons.long_name = 'longitude'
            lons.units = 'degrees_east'
            lons._CoordinateAxisType = "Lon"  # noqa W0212

            dimensions = ('lat', 'lon')
        else:
            dataset.createDimension('x', width)
            dataset.createDimension('y', height)

            lats = dataset.createVariable('y', 'f8', dimensions=('y'))
            lons = dataset.createVariable('x', 'f8', dimensions=('x'))

            lats.standard_name = 'projection_y_coordinate'
            lats.long_name = 'Northing'
            # TODO: How do we know if it's meters or something else?
            # lats.units = 'meters'
            lats._CoordinateAxisType = "GeoY"

            lons.standard_name = 'projection_x_coordinate'
            lons.long_name = "Easting"
            lons._CoordinateAxisType = "GeoX"

            dimensions = 'y', 'x'

        crs = dataset.createVariable('spatial_ref', 'i4')
        crs.spatial_ref = self.crs.wkt

        # Transform the cell indices to lat/lon based on the image crs
        # and transform
        x_coords, _ = rasterio.transform.xy(self.transform, np.zeros(width),
                                            np.arange(width))
        _, y_coords = rasterio.transform.xy(self.transform, np.arange(height),
                                            np.zeros(height))

        lons[:] = x_coords
        lats[:] = y_coords

        return dimensions
