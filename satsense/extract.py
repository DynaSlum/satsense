"""Module for computing features."""
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import groupby
from os import cpu_count

import numpy as np

from .image import FeatureVector

logger = logging.getLogger(__name__)


def extract_features_parallel(features, generator, n_jobs=cpu_count()):
    """Extract features in parallel."""
    logger.info("Extracting features using at most %s processes", n_jobs)
    generator.image.precompute_normalization()

    # Split generator in chunks
    generators = tuple(generator.split(n_chunks=n_jobs))

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for feature in features:
            extract = partial(extract_feature, feature)
            vector = np.ma.vstack(tuple(executor.map(extract, generators)))
            yield FeatureVector(feature, vector)


def extract_features(features, generator):
    """Compute features."""
    generator.image.precompute_normalization()

    for itype, group in groupby(features, lambda f: f.base_image):
        group = list(group)
        logger.info("Loading base image %s", itype)
        window_shapes = {
            shape
            for feature in group for shape in feature.windows
        }
        generator.load_image(itype, window_shapes)
        for feature in group:
            vector = extract_feature(feature, generator)
            yield FeatureVector(feature, vector)


def extract_feature(feature, generator):
    """Compute a feature vector."""
    logger.info("Computing feature %s with windows %s and arguments %s",
                feature.__class__.__name__, feature.windows, feature.kwargs)
    if not generator.loaded_itype == feature.base_image:
        logger.info("Loading base image %s", feature.base_image)
        generator.load_image(feature.base_image, feature.windows)

    shape = generator.shape + (len(feature.windows), feature.size)
    vector = np.ma.zeros((np.prod(shape[:-1]), feature.size), dtype=np.float32)
    vector.mask = np.zeros_like(vector, dtype=bool)

    size = vector.shape[0]
    for i, window in enumerate(generator):
        if window.shape[:2] not in feature.windows:
            continue
        if i % (size // 10 or 1) == 0:
            logger.info("%s%% ready", 100 * i // size)
        if window.mask.any():
            vector.mask[i] = True
        else:
            vector[i] = feature(window)

    vector.shape = shape
    return vector
