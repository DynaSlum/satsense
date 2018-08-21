"""Module for computing features."""
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from os import cpu_count

import numpy as np
import rasterio

from netCDF4 import Dataset

logger = logging.getLogger(__name__)


def extract_features_parallel(features, generator, n_jobs=cpu_count()):
    """Extract features in parallel."""
    logger.debug("Extracting features using at most %s processes", n_jobs)

    # Specify jobs
    generators = generator.split(n_jobs=n_jobs, features=features)

    # Execute jobs
    with ProcessPoolExecutor() as executor:
        extract = partial(extract_features, features)
        feature_vector = np.vstack(executor.map(extract, generators))

    logger.debug("Done extracting features. Feature vector shape %s",
                 feature_vector.shape)
    return feature_vector


def extract_features(features, generator):
    """Compute features."""
    shape = (generator.shape[0], generator.shape[1], features.index_size)
    feature_vector = np.zeros(
        (shape[0] * shape[1], shape[2]), dtype=np.float32)
    logger.debug("Feature vector shape %s", shape)

    # Pre compute images
    logger.debug("Using base images: %s", ', '.join(features.base_images))
    generator.image.precompute(features.base_images)

    size = len(generator)
    for i, cell in enumerate(generator):
        if i % (size // 10 or 1) == 0:
            logger.debug("%s%% ready", 100 * i // size)
        for feature in features.items.values():
            feature_vector[i, feature.indices] = feature(cell)
    feature_vector.shape = shape

    return feature_vector


def save_features(features,
                  feature_vector,
                  filename_prefix='',
                  extension='.nc',
                  crs=None,
                  transform=None):
    """Save computed features."""
    for name, feature in features.items.items():
        filename = filename_prefix + name + extension
        logger.debug("Saving feature %s to file %s", name, filename)
        data = feature_vector[:, :, feature.indices]
        if extension.lower() == '.nc':
            _save_array_as_netcdf(data, filename, name)
        elif extension.lower() == '.tif':
            _save_array_as_tif(data, filename, crs=crs, transform=transform)


def _save_array_as_netcdf(data, filename, feature_name):
    """Save feature array as NetCDF file."""
    width, height, length = data.shape
    with Dataset(filename, 'w') as dataset:
        dataset.createDimension('width', width)
        dataset.createDimension('height', height)
        dataset.createDimension('length', length)
        variable = dataset.createVariable(
            feature_name, 'f4', dimensions=('width', 'height', 'length'))
        variable[:] = data


def _save_array_as_tif(data, filename, crs, transform):
    """Save feature array as GeoTIFF file."""
    width, height, length = data.shape
    data = np.ma.filled(data)
    fill_value = data.fill_value if np.ma.is_masked(data) else None
    data = np.moveaxis(data, source=2, destination=0)
    with rasterio.open(
            filename,
            mode='w',
            driver='GTiff',
            width=width,
            height=height,
            count=length,
            dtype=rasterio.float32,
            crs=crs,
            transform=transform,
            nodata=fill_value) as dataset:
        dataset.write(data)


def load_features(features, filename_prefix):
    """Restore saved features."""
    feature_vector = None
    for name, feature in features.items.items():
        filename = filename_prefix + name + '.nc'
        logger.debug("Loading feature %s from file %s", name, filename)
        with Dataset(filename, 'r') as dataset:
            if feature_vector is None:
                shape = dataset.variables[name].shape[:2] + (
                    features.index_size, )
                feature_vector = np.empty(shape, dtype='f4')
            feature_vector[:, :, feature.indices] = dataset.variables[name][:]

    return feature_vector
