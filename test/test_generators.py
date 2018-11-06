import hypothesis.strategies as st
import numpy as np
import rasterio
from hypothesis import given
from hypothesis.control import assume
from hypothesis.extra.numpy import arrays

from satsense.bands import BANDS
from satsense.generators import FullGenerator
from satsense.image import Image

from .strategies import n_jobs, rasterio_dtypes


def create_test_file(filename, array):
    """Write an array of shape (bands, width, heigth) to file."""
    array = np.ma.asanyarray(array)
    with rasterio.open(
            filename,
            mode='w',
            driver='GTiff',
            width=array.shape[1],
            height=array.shape[2],
            count=array.shape[0],
            dtype=array.dtype) as dataset:
        for band, data in enumerate(array, start=1):
            dataset.write(data, band)


def create_test_image(dirname, array, normalization=None):
    """Create a test Image instance."""
    filename = str(dirname / 'tmp.tif')
    create_test_file(filename, array)
    satellite = 'quickbird'
    image = Image(filename, satellite, normalization_parameters=normalization)
    return image


def test_full_generator_windows(tmpdir):

    image_shape = (5, 5)
    window_shapes = ((5, 5), )
    step_size = (3, 3)
    satellite = 'quickbird'
    itype = 'grayscale'

    n_bands = len(BANDS[satellite])
    shape = (n_bands, ) + image_shape
    array = np.array(range(np.prod(shape)), dtype=float)
    array.shape = shape

    image = create_test_image(tmpdir, array, normalization=False)
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

step_size = st.tuples(
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=10),
)

image_array = arrays(
    dtype=rasterio_dtypes,
    shape=st.tuples(
        st.just(len(BANDS['quickbird'])),
        st.integers(min_value=2, max_value=10),
        st.integers(min_value=2, max_value=10),
    ),
)


@given(window_shapes, step_size, image_array)
def test_full_generator(tmpdir, window_shapes, step_size, image_array):
    # Assume that the image size is larger than the step size
    assume(image_array.shape[1] >= step_size[0])
    assume(image_array.shape[2] >= step_size[1])

    image = create_test_image(tmpdir, image_array, normalization=False)
    generator = FullGenerator(image, step_size)
    itype = 'grayscale'
    generator.load_image(itype, window_shapes)
    assert generator.loaded_itype == itype

    windows = []
    for window in generator:
        assert window.shape in window_shapes
        windows.append(window)
    assert np.prod(generator.shape) == len(windows) // len(window_shapes)


@given(window_shapes, step_size, image_array, n_jobs)
def test_full_generator_split(tmpdir, window_shapes, step_size, image_array,
                              n_jobs):
    # Assume that the image size is larger than the step size
    assume(image_array.shape[1] >= step_size[0])
    assume(image_array.shape[2] >= step_size[1])

    image = create_test_image(tmpdir, image_array, normalization=False)
    generator = FullGenerator(image, step_size)
    itype = 'grayscale'
    generator.load_image(itype, window_shapes)
    reference = list(generator)

    windows = []
    for gen in generator.split(n_jobs):
        gen.load_image(itype, window_shapes)
        windows.extend(gen)

    for i, window in enumerate(windows):
        np.testing.assert_array_equal(reference[i].mask, window.mask)
        np.testing.assert_array_equal(reference[i][~reference[i].mask],
                                      window[~window.mask])

    assert len(reference) == len(windows)
