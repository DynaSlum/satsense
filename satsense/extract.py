"""Module for computing features."""
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from os import cpu_count

import numpy as np
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
    itypes = {f.base_image for f in features.items.values()}
    logger.debug("Using base images: %s", ', '.join(itypes))
    for itype in itypes:
        getattr(generator.image, itype)

    size = len(generator)
    for i, cell in enumerate(generator):
        if i % (size // 10 or 1) == 0:
            logger.debug("%s%% ready", 100 * i // size)
        for feature in features.items.values():
            feature_vector[i, feature.indices] = feature(cell)
    feature_vector.shape = shape

    return feature_vector


def save_features(features, feature_vector, filename_prefix=''):
    """Save computed features."""
    for name, feature in features.items.items():
        filename = filename_prefix + name + '.nc'
        logger.debug("Saving feature %s to file %s", name, filename)
        data = feature_vector[:, :, feature.indices]
        with Dataset(filename, 'w') as dataset:
            size_y, size_x, size_feature = data.shape
            dataset.createDimension('y', size_y)
            dataset.createDimension('x', size_x)
            dataset.createDimension('feature', size_feature)
            variable = dataset.createVariable(
                name, 'f4', dimensions=('y', 'x', 'feature'))
            variable[:] = data
