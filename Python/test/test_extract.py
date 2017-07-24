from satsense import SatelliteImage
from satsense.generators import CellGenerator
from satsense.extract import extract_features
from satsense.features import FeatureSet, Pantex



# Supported image formats include RGB, Quickbird and Worldview
from satsense  import RGB, QUICKBIRD, WORLDVIEW2
import numpy as np
import pytest

def load_image():
    # URI to the image
    imagefile = '/home/bweel/Documents/projects/dynaslum/data/satelite/056239125010_01/056239125010_01_P001_MUL/08NOV02054348-M2AS_R1C1-056239125010_01_P001.TIF'
    # Set the correct format here, it is used throughout the notebook
    bands = QUICKBIRD

    # Loading the file
    image = SatelliteImage.load_from_file(imagefile, bands)

    return image


def test_generator():
    image = load_image()

    generator = CellGenerator(image, (25, 25), length=(2, 5))
    for cell in generator:
        assert cell


def test_extract_features():
    image = load_image()
    generator = CellGenerator(image, (25, 25), length=(5, 10))

    features = FeatureSet()
    features.add(Pantex(windows=((25, 25), (50, 50), (100, 100))))

    results = extract_features(features, generator)

    assert results.any()
