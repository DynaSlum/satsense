from os import cpu_count

import numpy as np

from satsense.features import FeatureSet, HistogramOfGradients
from satsense.generators import CellGenerator
from satsense.image import Image

from test.test_extract import image


def test_generator(image):

    generator = CellGenerator(image, (25, 25))
    assert len(generator) == len(tuple(generator))


def test_padding(image):

    generator = CellGenerator(image, (25, 25), length=(3, 2))
    for cell in generator:
        assert cell.shape == (25, 25)
        assert cell.super_cell((100, 100)).shape == (100, 100)


def assert_image_equivalent(img: Image, other: Image):
    assert img.bands == other.bands
    assert img._normalization_parameters == other._normalization_parameters
    for itype in img._images:
        np.testing.assert_array_almost_equal_nulp(img._images[itype],
                                                  getattr(other, itype))


def test_generator_split():

    cell_size = (10, 10)
    windows = [(25, 25), (50, 50)]

    features = FeatureSet()
    features.add(HistogramOfGradients(windows=windows))

    reference = tuple(CellGenerator(image(), cell_size))
    generators = CellGenerator(image(), cell_size).split(
        n_jobs=cpu_count(), features=features)

    cells = []
    i = 0
    for generator in generators:
        for cell in generator:
            ref = reference[i]
            print(cell.x, cell.y, ref.x, ref.y)
            for window in windows:
                assert_image_equivalent(
                    cell.super_cell(window), ref.super_cell(window))
            cells.append(cell)
            i += 1

    assert len(reference) == len(cells)
