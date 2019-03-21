import hypothesis.strategies as st
import numpy as np
import pytest
import rasterio
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from rasterio.transform import from_origin

from satsense.bands import BANDS
from satsense.generators import FullGenerator
from satsense.image import Image

from .strategies import st_rasterio_dtypes


def create_test_file(filename, array):
    """Write an array of shape (bands, width, height) to file."""
    array = np.ma.asanyarray(array)
    crs = rasterio.crs.CRS(init='epsg:4326')
    transform = from_origin(52, 4, 10, 10)
    with rasterio.open(
            filename,
            mode='w',
            driver='GTiff',
            count=array.shape[0],
            height=array.shape[1],
            width=array.shape[2],
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


def create_mono_image(dirname, array, normalization=None):
    """Create a test Image instance."""
    filename = str(dirname / 'tmp.tif')
    create_test_file(filename, array)
    satellite = 'monochrome'
    image = Image(filename, satellite, normalization_parameters=normalization)
    return image


WINDOW_TEST_DATA = [
    {
        'image_shape': (3, 3),
        'step_size': (2, 1),
        'window_shape': (3, 4),
        'window_ref_arrays': [
            np.ma.array(
                [
                    [0, 0, 0, 1],
                    [0, 0, 3, 4],
                    [0, 0, 6, 7],
                ],
                mask=[
                    [True, True, False, False],
                    [True, True, False, False],
                    [True, True, False, False],
                ]
            ),
            np.ma.array(
                [
                    [0, 0, 1, 2],
                    [0, 3, 4, 5],
                    [0, 6, 7, 8],
                ],
                mask=[
                    [True, False, False, False],
                    [True, False, False, False],
                    [True, False, False, False],
                ]
            ),
            np.ma.array(
                [
                    [0, 1, 2, 0],
                    [3, 4, 5, 0],
                    [6, 7, 8, 0],
                ],
                mask=[
                    [False, False, False, True],
                    [False, False, False, True],
                    [False, False, False, True],
                ]
            ),
            np.ma.array(
                [
                    [0, 0, 6, 7],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                mask=[
                    [True, True, False, False],
                    [True, True, True, True],
                    [True, True, True, True],
                ]
            ),
            np.ma.array(
                [
                    [0, 6, 7, 8],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                mask=[
                    [True, False, False, False],
                    [True, True, True, True],
                    [True, True, True, True],
                ]
            ),
            np.ma.array(
                [
                    [6, 7, 8, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                mask=[
                    [False, False, False, True],
                    [True, True, True, True],
                    [True, True, True, True],
                ]
            ),
        ]
    },
    {
        'image_shape': (6, 6),
        'step_size': (2, 3),
        'window_shape': (5, 5),
        'window_ref_arrays': [
            np.ma.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 2, 3],
                    [0, 6, 7, 8, 9],
                    [0, 12, 13, 14, 15],
                    [0, 18, 19, 20, 21],
                ],
                mask=[
                    [True, True, True, True, True],
                    [True, False, False, False, False],
                    [True, False, False, False, False],
                    [True, False, False, False, False],
                    [True, False, False, False, False],
                ]
            ),
            np.ma.array(
                [
                    [0, 0, 0, 0, 0],
                    [2, 3, 4, 5, 0],
                    [8, 9, 10, 11, 0],
                    [14, 15, 16, 17, 0],
                    [20, 21, 22, 23, 0],
                ],
                mask=[
                    [True, True, True, True, True],
                    [False, False, False, False, True],
                    [False, False, False, False, True],
                    [False, False, False, False, True],
                    [False, False, False, False, True],
                ]
            ),
            np.ma.array(
                [
                    [0, 6, 7, 8, 9],
                    [0, 12, 13, 14, 15],
                    [0, 18, 19, 20, 21],
                    [0, 24, 25, 26, 27],
                    [0, 30, 31, 32, 33],
                ],
                mask=[
                    [True, False, False, False, False],
                    [True, False, False, False, False],
                    [True, False, False, False, False],
                    [True, False, False, False, False],
                    [True, False, False, False, False],
                ]
            ),
            np.ma.array(
                [
                    [8, 9, 10, 11, 0],
                    [14, 15, 16, 17, 0],
                    [20, 21, 22, 23, 0],
                    [26, 27, 28, 29, 0],
                    [32, 33, 34, 35, 0],
                ],
                mask=[
                    [False, False, False, False, True],
                    [False, False, False, False, True],
                    [False, False, False, False, True],
                    [False, False, False, False, True],
                    [False, False, False, False, True],
                ]
            ),
            np.ma.array(
                [
                    [0, 18, 19, 20, 21],
                    [0, 24, 25, 26, 27],
                    [0, 30, 31, 32, 33],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                mask=[
                    [True, False, False, False, False],
                    [True, False, False, False, False],
                    [True, False, False, False, False],
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                ]
            ),
            np.ma.array(
                [
                    [20, 21, 22, 23, 0],
                    [26, 27, 28, 29, 0],
                    [32, 33, 34, 35, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                mask=[
                    [False, False, False, False, True],
                    [False, False, False, False, True],
                    [False, False, False, False, True],
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                ]
            )
        ],
    },
    {
        'image_shape': (4, 6),
        'step_size': (2, 3),
        'window_shape': (5, 2),
        'window_ref_arrays': [
            np.ma.array(
                [
                    [0, 0],
                    [0, 1],
                    [6, 7],
                    [12, 13],
                    [18, 19],
                ],
                mask=[
                    [True, True],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                ]
            ),
            np.ma.array(
                [
                    [0, 0],
                    [3, 4],
                    [9, 10],
                    [15, 16],
                    [21, 22],
                ],
                mask=[
                    [True, True],
                    [False, False],
                    [False, False],
                    [False, False],
                    [False, False],
                ]
            ),
            np.ma.array(
                [
                    [6, 7],
                    [12, 13],
                    [18, 19],
                    [0, 0],
                    [0, 0],
                ],
                mask=[
                    [False, False],
                    [False, False],
                    [False, False],
                    [True, True],
                    [True, True],
                ]
            ),
            np.ma.array(
                [
                    [9, 10],
                    [15, 16],
                    [21, 22],
                    [0, 0],
                    [0, 0],
                ],
                mask=[
                    [False, False],
                    [False, False],
                    [False, False],
                    [True, True],
                    [True, True],
                ]
            )
        ]
    }
]


@pytest.mark.parametrize(list(WINDOW_TEST_DATA[0]),
                         [tuple(d.values()) for d in WINDOW_TEST_DATA])
def test_windows(tmpdir, image_shape, step_size, window_shape,
                 window_ref_arrays):
    satellite = 'monochrome'
    itype = 'pan'

    n_bands = len(BANDS[satellite])
    shape = (n_bands, ) + image_shape
    image_array = np.array(range(np.prod(shape)), dtype=float)
    image_array.shape = shape
    print('image_array= ', image_array)
    image = create_mono_image(tmpdir, image_array, normalization=False)
    generator = FullGenerator(image, step_size)
    generator.load_image(itype, (window_shape, ))
    windows = [window for window in generator]
    print('generator._image_cache:\n', generator._image_cache)

    assert len(window_ref_arrays) == len(windows)
    for window, reference in zip(windows, window_ref_arrays):
        np.testing.assert_array_equal(window, reference)
        assert np.all(window.mask == reference.mask)


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

st_window_shapes = st.lists(
    st_window_shape, min_size=1, max_size=10, unique=True)


def create_step_and_image_strategy(limit, dtype):
    step_size = st.tuples(
        st.integers(min_value=1, max_value=limit[0]),
        st.integers(min_value=1, max_value=limit[1]),
    )

    image_array = arrays(
        dtype=dtype,
        shape=st.tuples(
            st.just(len(BANDS['quickbird'])),
            st.integers(min_value=limit[0], max_value=10),
            st.integers(min_value=limit[1], max_value=10),
        ),
    )

    return st.tuples(step_size, image_array)


st_step_and_image = st.tuples(
    st.tuples(
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=10),
    ),
    st_rasterio_dtypes,
).flatmap(lambda args: create_step_and_image_strategy(*args))


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
@settings(deadline=1000)
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
