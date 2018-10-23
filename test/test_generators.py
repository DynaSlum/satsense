import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from hypothesis.extra.numpy import arrays, scalar_dtypes

from satsense.features import FeatureSet, HistogramOfGradients
from satsense.generators import FullGenerator
from satsense.image import Image


@pytest.fixture
def image(monkeypatch):
    """Create a test Image instance."""
    monkeypatch.setattr(Image, 'shape', (10, 10))

    def _read_band(self, band, block=None):
        image = np.ma.array(range(np.prod(self.shape)))
        image.shape = self.shape
        if block is None:
            return image

        shape = tuple(end - start for start, end in block)
        padded_image = np.ma.empty(shape)
        padded_image.mask = np.ones(shape, dtype=bool)
        # TODO: fix inserting image in padded image

        return padded_image

    monkeypatch.setattr(Image, '_read_band', _read_band)

    image = Image('filename', 'quickbird')
    image.precompute_normalization()

    return image


@pytest.fixture
def generator(image):
    """Create a test FullGenerator instance."""
    step_size = (2, 3)
    generator = FullGenerator(image, step_size)
    return generator


window_shape = st.tuples(
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=10))
window_shapes = st.lists(window_shape, min_size=1, max_size=10)


@given(window_shapes)
def test_generator(generator, window_shapes):

    itype = 'grayscale'
    generator.load_image(itype, window_shapes)

    windows = []
    for window in generator:
        assert window.shape in window_shapes
        windows.append(window)
    assert np.prod(generator.shape) == len(windows) // len(window_shapes)


@given(window_shapes, st.integers(min_value=1, max_value=10))
def test_generator_split(generator, window_shapes, n_jobs):

    itype = 'grayscale'
    generator.load_image(itype, window_shapes)
    reference = list(generator)

    windows = []
    for gen in generator.split(n_jobs):
        gen.load_image(itype, window_shapes)
        windows.extend(gen)

    for i, window in enumerate(windows):
        np.testing.assert_array_equal(reference[i], window)

    assert len(reference) == len(windows)
