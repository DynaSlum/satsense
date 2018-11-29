"""Test feature extraction related functions."""
import os

import numpy as np
import pytest
from hypothesis import given

from satsense.bands import BANDS
from satsense.extract import extract_features
from satsense.features import HistogramOfGradients, NirNDVI, Pantex
from satsense.generators import FullGenerator
from satsense.image import FeatureVector

from .strategies import st_n_jobs
from .test_generators import create_test_image

@pytest.fixture
def generator(tmpdir):
    image_shape = (10, 10)
    step_size = (3, 3)
    satellite = 'worldview3'

    n_bands = len(BANDS[satellite])
    shape = (n_bands, ) + image_shape
    array = np.array(range(np.prod(shape)), dtype=float)
    array.shape = shape
    image = create_test_image(tmpdir, array)
    generator = FullGenerator(image, step_size)
    return generator


def test_save_load_roundtrip(tmpdir, generator):
    """Test that saving and loading does not modify a FeatureVector."""
    window_shapes = ((3, 3), (5, 5))

    feature = Pantex(window_shapes)

    shape = (*generator.shape, len(window_shapes), feature.size)
    vector = np.array(range(np.prod(shape)), dtype=float)
    vector.shape = shape

    feature_vector = FeatureVector(feature, vector, crs=generator.crs, transform=generator.transform)
    prefix = str(tmpdir) + os.sep
    feature_vector.save(prefix)
    restored_vector = feature_vector.from_file(feature, prefix)

    np.testing.assert_array_almost_equal_nulp(feature_vector.vector,
                                              restored_vector.vector)


def test_extract_features(generator):
    """Test that features can be computed."""
    window_shapes = (
        (3, 3),
        (5, 5),
    )
    feature = Pantex(window_shapes)

    result = list(extract_features([feature], generator, n_jobs=1))[0]

    assert result.feature == feature
    assert result.vector.any()
    assert result.vector.shape == generator.shape + (len(window_shapes),
                                                     feature.size)


@given(st_n_jobs)
def test_extract_features_parallel(generator, n_jobs):
    """Test that parallel feature computation produces identical results."""
    window_shapes = (
        (3, 3),
        (5, 5),
    )
    hog = HistogramOfGradients(window_shapes)
    ndvi = NirNDVI(window_shapes)
    features = [hog, ndvi]
    print("Computing reference features")
    references = list(extract_features(features, generator, n_jobs=1))
    print("Computing features in parallel")
    results = list(extract_features(features, generator, n_jobs=n_jobs))

    for reference, result in zip(references, results):
        np.testing.assert_array_equal(result.vector.mask,
                                      reference.vector.mask)
        np.testing.assert_array_almost_equal_nulp(
            result.vector[~result.vector.mask],
            reference.vector[~reference.vector.mask])
