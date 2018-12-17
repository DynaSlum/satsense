import numpy as np
import rasterio
from hypothesis import given
from netCDF4 import Dataset

from satsense.image import FeatureVector

from .test_extract import RGBFeature
from .test_generators import st_window_shape as st_image_shape
from .test_generators import st_window_shapes


def create_featurevector(image_shape, window_shapes):
    """Create a FeatureVector instance for testing."""
    feature = RGBFeature(window_shapes, test=True)
    vector = np.ma.empty((*image_shape, len(feature.windows), feature.size))
    vector.mask = np.zeros_like(vector, dtype=bool)
    for i in range(len(feature.windows)):
        for j in range(feature.size):
            vector[:, :, i, j] = 100 * i + j
            vector.mask[0, 0, i, j] = True
    crs = rasterio.crs.CRS(init='epsg:4326')
    transform = rasterio.transform.from_origin(52, 4, 10, 10)
    featurevector = FeatureVector(feature, vector, crs, transform)
    return feature, vector, featurevector


@given(st_image_shape, st_window_shapes)
def test_save_as_netcdf(tmpdir, image_shape, window_shapes):
    feature, vector, featurevector = create_featurevector(
        image_shape, window_shapes)
    filenames = featurevector.save(tmpdir, extension='nc')

    assert len(filenames) == len(feature.windows)
    for i, (filename, window_shape) in enumerate(
            zip(filenames, feature.windows)):
        with Dataset(filename) as dataset:
            assert tuple(dataset.window) == window_shape
            assert dataset.arguments == repr(feature.kwargs)
            data = dataset.variables[feature.name][:]
            assert data.shape == (feature.size, ) + vector.shape[:2]
            for j in range(feature.size):
                try:
                    np.testing.assert_equal(data.mask[j],
                                            vector.mask[..., i, j])
                    np.testing.assert_array_almost_equal_nulp(
                        data[j].compressed(), vector[..., i, j].compressed())
                except AssertionError:
                    print("Error in window", i, "feature element", j)
                    print('saved:\n', data[j])
                    print('reference:\n', vector[..., i, j])
                    raise


@given(st_image_shape, st_window_shapes)
def test_save_as_geotiff(tmpdir, image_shape, window_shapes):
    feature, vector, featurevector = create_featurevector(
        image_shape, window_shapes)
    filenames = featurevector.save(tmpdir, extension='tif')

    assert len(filenames) == len(feature.windows)
    for i, (filename, window_shape) in enumerate(
            zip(filenames, feature.windows)):
        with rasterio.open(filename) as dataset:
            assert dataset.tags()['window'] == repr(window_shape)
            assert dataset.tags()['arguments'] == repr(feature.kwargs)
            assert dataset.shape == vector.shape[:2]
            assert dataset.count == feature.size
            for j in range(feature.size):
                band = j + 1
                data = dataset.read(band, masked=True)
                try:
                    np.testing.assert_equal(data.mask, vector.mask[..., i, j])
                    np.testing.assert_array_almost_equal_nulp(
                        data.compressed(), vector[..., i, j].compressed())
                except AssertionError:
                    print("Error in window", i, "feature element", j)
                    print('saved:\n', data)
                    print('reference:\n', vector[..., i, j])
                    raise
