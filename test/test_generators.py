import hypothesis.strategies as st
import numpy as np
import rasterio
from hypothesis import given
from hypothesis.extra.numpy import arrays
from rasterio.transform import from_origin

from satsense.bands import BANDS
from satsense.generators import FullGenerator
from satsense.image import Image

from .strategies import st_rasterio_dtypes


def create_test_file(filename, array):
    """Write an array of shape (bands, width, heigth) to file."""
    array = np.ma.asanyarray(array)
    crs = rasterio.crs.CRS(init='epsg:4326')
    transform = from_origin(52, 4, 10, 10)
    with rasterio.open(
            filename,
            mode='w',
            driver='GTiff',
            width=array.shape[1],
            height=array.shape[2],
            count=array.shape[0],
            dtype=array.dtype,
            crs=crs,
            transform=transform) as dataset:
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


st_window_shape = st.tuples(
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=10))

st_window_shapes = st.lists(st_window_shape, min_size=1, max_size=10)


def create_step_and_image_strategy(limit):

    step_size = st.tuples(
        st.integers(min_value=1, max_value=limit[0]),
        st.integers(min_value=1, max_value=limit[1]),
    )

    image_array = arrays(
        dtype=st_rasterio_dtypes,
        shape=st.tuples(
            st.just(len(BANDS['quickbird'])),
            st.integers(min_value=limit[0], max_value=10),
            st.integers(min_value=limit[1], max_value=10),
        ),
    )

    return st.tuples(step_size, image_array)


st_step_and_image = st.tuples(
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=10),
).flatmap(create_step_and_image_strategy)


@given(st_window_shapes, st_step_and_image)
def test_full_generator(tmpdir, window_shapes, step_and_image):
    step_size, image_array = step_and_image

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


@given(st_window_shapes, st_step_and_image,
       st.integers(min_value=1, max_value=5))
def test_full_generator_split(tmpdir, window_shapes, step_and_image, n_chunks):
    step_size, image_array = step_and_image

    image = create_test_image(tmpdir, image_array, normalization=False)
    generator = FullGenerator(image, step_size)
    itype = 'grayscale'
    generator.load_image(itype, window_shapes)
    reference = list(generator)

    windows = []
    for gen in generator.split(n_chunks):
        gen.load_image(itype, window_shapes)
        windows.extend(gen)

    for i, window in enumerate(windows):
        np.testing.assert_array_equal(reference[i].mask, window.mask)
        np.testing.assert_array_equal(reference[i][~reference[i].mask],
                                      window[~window.mask])

    assert len(reference) == len(windows)
