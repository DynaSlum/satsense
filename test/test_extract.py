"""Test feature extraction related functions."""
import os

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

from satsense.bands import BANDS
from satsense.extract import extract_features
from satsense.features import Feature
from satsense.generators import FullGenerator
from satsense.image import FeatureVector

from .test_generators import create_test_image


class BaseTestFeature(Feature):
    size = 1
    compute = staticmethod(lambda a: np.mean(a, axis=(0, 1)))


class GrayscaleFeature(BaseTestFeature):
    base_image = 'grayscale'


class GrayUbyteFeature(BaseTestFeature):
    base_image = 'gray_ubyte'


class RGBFeature(BaseTestFeature):
    base_image = 'rgb'
    size = 3


def test_save_load_roundtrip(tmpdir):
    """Test that saving and loading does not modify a FeatureVector."""
    window_shapes = ((3, 3), (5, 5))

    feature = GrayscaleFeature(window_shapes)

    shape = (2, 3, len(window_shapes), feature.size)
    vector = np.array(range(np.prod(shape)), dtype=float)
    vector.shape = shape

    feature_vector = FeatureVector(feature, vector)
    prefix = str(tmpdir) + os.sep
    feature_vector.save(prefix)
    restored_vector = feature_vector.from_file(feature, prefix)

    np.testing.assert_array_almost_equal_nulp(feature_vector.vector,
                                              restored_vector.vector)


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


def test_extract_features(generator):
    """Test that features can be computed."""
    features = [
        RGBFeature(window_shapes=((3, 2), )),
        GrayUbyteFeature(window_shapes=(
            (3, 2),
            (4, 4),
        )),
        GrayUbyteFeature(window_shapes=(
            (4, 4),
            (2, 5),
        ))
    ]

    results = tuple(extract_features(features, generator, n_jobs=1))

    assert len(results) == 3
    for result, feature in zip(results, features):
        assert result.feature == feature
        assert result.vector.any()
        shape = generator.shape + (len(feature.windows), feature.size)
        assert result.vector.shape == shape


@given(st.integers(min_value=-1, max_value=10))
def test_extract_features_parallel(generator, n_jobs):
    """Test that parallel feature computation produces identical results."""
    window_shapes = (
        (3, 3),
        (5, 5),
    )
    features = [
        GrayscaleFeature(window_shapes),
    ]
    print("Computing reference features")
    references = list(extract_features(features, generator, n_jobs=1))
    assert len(references) == 1

    print("Computing features in parallel")
    results = list(extract_features(features, generator, n_jobs=n_jobs))
    assert len(results) == 1

    for reference, result in zip(references, results):
        np.testing.assert_array_equal(result.vector.mask,
                                      reference.vector.mask)
        np.testing.assert_array_almost_equal_nulp(
            result.vector[~result.vector.mask],
            reference.vector[~reference.vector.mask])
