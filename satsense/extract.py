"""Module for computing features."""
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import groupby
from os import cpu_count
from typing import Iterator

import numpy as np

from .features import Feature
from .generators import FullGenerator
from .image import FeatureVector

logger = logging.getLogger(__name__)


def extract_features(features: Iterator[Feature],
                     generator: FullGenerator,
                     n_jobs: int = -1):
    """Compute features.

    Parameters
    ----------
    features:
        Iterable of features.
    generator:
        Generator providing the required windows on the image.
    n_jobs:
        The maximum number of processes to use. The default is to use the
        value returned by :func:`os.cpu_count`.

    Yields
    ------
    :obj:`satsense.FeatureVector`
        The requested feature vectors.

    Examples
    --------
    Extracting features from an image::

        import numpy as np
        from satsense import Image
        from satsense.generators import FullGenerator
        from satsense.extract import extract_features
        from satsense.features import NirNDVI, HistogramOfGradients, Pantex

        # Define the features to calculate
        features = [
            HistogramOfGradients(((50, 50), (100, 100))),
            NirNDVI(((50, 50),)),
            Pantex(((50, 50), (100, 100))),
        ]

        # Load the image into a generator
        # This generator splits the image into chunks of 10x10 pixels
        image = Image('test/data/source/section_2_sentinel.tif', 'quickbird')
        image.precompute_normalization()
        generator = FullGenerator(image, (10, 10))

        # Calculate all the features and append them to a list
        vector = []
        for feature_vector in extract_features(features, generator):
            # The shape returned is (x, y, w, v)
            # where x is the number of chunks in the x direction
            #       y is the number of chunks in the y direction
            #       w is the number of windows the feature uses
            #       v is the length of the feature per window
            # Reshape the resulting vector so it is (x, y, w * v)
            # e.g. flattened along the windows and features
            data = feature_vector.vector.reshape(
                        *feature_vector.vector.shape[0:2], -1)
            vector.append(data)
        # dstack reshapes the vector into and (x, y, n)
        # where n is the total length of all features
        featureset = np.dstack(vector)
    """
    if n_jobs == 1:
        yield from _extract_features(features, generator)
    else:
        yield from _extract_features_parallel(features, generator, n_jobs)


def _extract_features_parallel(features, generator, n_jobs=-1):
    """Extract features in parallel."""
    if n_jobs < 1:
        n_jobs = cpu_count()
    logger.info("Extracting features using at most %s processes", n_jobs)
    generator.image.precompute_normalization()

    # Split generator in chunks
    generators = tuple(generator.split(n_chunks=n_jobs))

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for feature in features:
            extract = partial(extract_feature, feature)
            vector = np.ma.vstack(tuple(executor.map(extract, generators)))
            yield FeatureVector(feature, vector, generator.crs,
                                generator.transform)


def _extract_features(features, generator):
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
            yield FeatureVector(feature, vector, generator.crs,
                                generator.transform)


def extract_feature(feature, generator):
    """Compute a single feature vector.

    Parameters
    ----------
    feature : Feature
        The feature to calculate
    generator:
        Generator providing the required windows on the image.
    """
    logger.info("Computing feature %s with windows %s and arguments %s",
                feature.__class__.__name__, feature.windows, feature.kwargs)
    if not generator.loaded_itype == feature.base_image:
        logger.info("Loading base image %s", feature.base_image)
        generator.load_image(feature.base_image, feature.windows)

    shape = generator.shape + (len(feature.windows), feature.size)
    vector = np.ma.zeros((np.prod(shape[:-1]), feature.size), dtype=np.float32)
    vector.mask = np.zeros_like(vector, dtype=bool)

    size = vector.shape[0]
    i = 0
    for window in generator:
        if window.shape[:2] not in feature.windows:
            continue
        if i % (size // 10 or 1) == 0:
            logger.info("%s%% ready", 100 * i // size)
        if window.mask.any():
            vector.mask[i] = True
        else:
            vector[i] = feature(window)
        i += 1

    vector.shape = shape
    return vector
