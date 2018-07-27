import numpy as np

import pytest
# Supported image formats include RGB, Quickbird and Worldview
from satsense import QUICKBIRD, SatelliteImage
from satsense.extract import extract_features, extract_features_parallel
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


def load_image():
    # URI to the image
    imagefile = '/home/bweel/Documents/projects/dynaslum/data/satelite/056239125010_01/056239125010_01_P001_MUL/08NOV02054348-M2AS_R1C1-056239125010_01_P001.TIF'
    # Set the correct format here, it is used throughout the notebook
    bands = QUICKBIRD

    # Loading the file
    image = SatelliteImage.load_from_file(imagefile, bands)

    return image


def test_extract_features(image):

    generator = CellGenerator(image, (10, 10), length=(10, 5))

    features = FeatureSet()
    features.add(Pantex(windows=((25, 25), (50, 50), (100, 100))))

    results = extract_features(features, generator)

    assert results.any()


@pytest.mark.parametrize("n_jobs", [1, 2, 3, 4, 5, 10])
def test_extract_features_parallel(image, n_jobs):
    """Test that parallel extraction produces identical results."""
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
