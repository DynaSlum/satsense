import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
# TODO: use modules below to generate test images?
# from hypothesis.extra.numpy import arrays, scalar_dtypes

from satsense.generators import FullGenerator
from satsense.image import Image


@pytest.fixture
def image(monkeypatch):
    """Create a test Image instance."""

    def _read_band(self, band, block=None):
        """Simulate the behaviour of rasterio padded reading."""
        dtype = np.float32
        image = np.ma.array(range(np.prod(self.shape)), dtype=dtype)
        image.shape = self.shape
        if block is None:
            return image

        # insert (part of) the image defined above in a padded image
        shape = tuple(end - start for start, end in block)
        padded_image = np.ma.empty(shape, dtype=dtype)
        padded_image.mask = np.ones(shape, dtype=bool)

        islice = []
        pslice = []
        for (start, end), ilen in zip(block, image.shape):
            if start < 0:
                # case: prepend -start pixels of padding
                istart = 0
                pstart = -start
            else:
                # case: skip first start pixels of image
                istart = start
                pstart = 0

            length = min(ilen, end) - istart

            islice.append(slice(istart, istart + length))
            pslice.append(slice(pstart, pstart + length))

        padded_image[pslice[0], pslice[1]] = image[islice[0], islice[1]]
        padded_image.mask[pslice[0], pslice[1]] = False

        return padded_image

    monkeypatch.setattr(Image, '_read_band', _read_band)

    monkeypatch.setattr(Image, 'shape', (5, 5))

    image = Image('filename', 'quickbird')
    image.precompute_normalization()

    return image


@pytest.fixture
def generator(image):
    """Create a test FullGenerator instance."""
    step_size = (2, 3)
    generator = FullGenerator(image, step_size)
    return generator


def test_generated_windows(image):

    assert image.shape == (5, 5), "Wrong test input shape"

    itype = 'gray_ubyte'
    window_shapes = ((5, 5), )
    step_size = (3, 3)

    generator = FullGenerator(image, step_size)
    generator.load_image(itype, window_shapes)

    print('generator._image_cache:\n', generator._image_cache)

    assert generator.offset == (0, 0)
    assert generator.shape == (2, 2)

    windows = [window for window in generator]
    assert len(windows) == 4

    for i, window in enumerate(windows):
        print('window', i, '\n', window)

    # window center pixels are correct
    image._block = None
    original_image = image[itype]
    print('original image:\n', original_image)
    assert windows[0][2][2] == original_image[1][1]
    assert windows[1][2][2] == original_image[1][4]
    assert windows[2][2][2] == original_image[4][1]
    assert windows[3][2][2] == original_image[4][4]

    # horizontal edges are masked
    assert np.all(windows[0].mask[0])
    assert np.all(windows[1].mask[0])
    assert np.all(windows[2].mask[-1])
    assert np.all(windows[3].mask[-1])

    # vertical edges are masked
    assert np.all(windows[0].mask[:, 0])
    assert np.all(windows[1].mask[:, 3:])
    assert np.all(windows[2].mask[:, 0])
    assert np.all(windows[3].mask[:, 3:])

    # data is not masked
    assert not np.any(windows[0].mask[1:, 1:])
    assert not np.any(windows[1].mask[1:, :3])
    assert not np.any(windows[2].mask[:3, 1:])
    assert not np.any(windows[3].mask[:3, :3])


window_shape = st.tuples(
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=10))
window_shapes = st.lists(window_shape, min_size=1, max_size=10)


@given(window_shapes)
def test_generator(generator, window_shapes):

    itype = 'grayscale'
    generator.load_image(itype, window_shapes)
    assert generator.loaded_itype == itype

    windows = []
    for window in generator:
        assert window.shape in window_shapes
        windows.append(window)
    assert np.prod(generator.shape) == len(windows) // len(window_shapes)


n_jobs = st.integers(min_value=1, max_value=10)


@given(window_shapes, n_jobs)
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
