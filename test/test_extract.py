"""Test feature extraction related functions."""
import os
import tempfile

import numpy as np
import pytest

# Supported image formats include RGB, Quickbird and Worldview
from satsense import QUICKBIRD, SatelliteImage
from satsense.extract import (extract_features, extract_features_parallel,
                              load_features, save_features)
from satsense.features import FeatureSet, HistogramOfGradients, Pantex
from satsense.generators import CellGenerator


@pytest.fixture
def image():
    """Create a test SatelliteImage instance."""
    bands = QUICKBIRD
    shape = (95, 50, len(bands))
    data = np.array(range(np.prod(shape)), dtype=np.float32)
    data.shape = shape
    return SatelliteImage(data, bands)


def test_save_load_roundtrip():
    """Test that saving and loading does not modify a feature array."""
    features = FeatureSet()
    features.add(HistogramOfGradients())
    features.add(Pantex())

    shape = (2, 3, features.index_size)
    feature_vector_in = np.array(range(np.prod(shape)), dtype=float)
    feature_vector_in.shape = shape

    with tempfile.TemporaryDirectory() as out_dir:
        prefix = out_dir + os.sep
        save_features(features, feature_vector_in, filename_prefix=prefix)
        feature_vector_out = load_features(features, filename_prefix=prefix)

    np.testing.assert_array_almost_equal_nulp(feature_vector_in,
                                              feature_vector_out)


def test_extract_features(image):
    """Test that features can be computed."""
    generator = CellGenerator(image, (10, 10), length=(10, 5))

    features = FeatureSet()
    features.add(Pantex(windows=((25, 25), (50, 50), (100, 100))))

    results = extract_features(features, generator)

    assert results.any()


@pytest.mark.parametrize("n_jobs", [1, 2, 3, 4, 5, 10])
def test_extract_features_parallel(image, n_jobs):
    """Test that parallel feature computation produces identical results."""
    cell_size = (10, 10)
    windows = [(25, 25), (50, 50)]

    features = FeatureSet()
    features.add(HistogramOfGradients(windows=windows))
    print("Computing reference features")
    generator = CellGenerator(image, cell_size)
    reference = extract_features(features, generator)
    print("Computing features in parallel")
    generator = CellGenerator(image, cell_size)
    results = extract_features_parallel(features, generator, n_jobs)

    np.testing.assert_array_almost_equal_nulp(results, reference)
