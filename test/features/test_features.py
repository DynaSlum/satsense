"""
Testing features
"""
import numpy as np
import pytest
import rasterio
from netCDF4 import Dataset
from sklearn.cluster import MiniBatchKMeans

from satsense.features.hog import hog_features
from satsense.features.lacunarity import lacunarities
from satsense.features.ndxi import ndxi_image
from satsense.features.pantex import pantex
from satsense.features.sift import sift, sift_cluster
from satsense.features.texton import (get_texton_descriptors, texton,
                                      texton_cluster)


def test_ndvi(image):
    """
    Test ndvi
    """
    with rasterio.open('test/data/target/ndvi.tif') as dataset:
        target = dataset.read(masked=True)

    result = ndxi_image(image, 'nir_ndvi')

    same = result == target

    assert same.all()


def test_rg_ndvi(image):
    """
    Test ndvi
    """
    with rasterio.open('test/data/target/rg_ndvi.tif') as dataset:
        target = dataset.read(masked=True)

    result = ndxi_image(image, 'rg_ndvi')

    same = result == target

    assert same.all()


def test_rb_ndvi(image):
    """
    Test ndvi
    """
    with rasterio.open('test/data/target/rb_ndvi.tif') as dataset:
        target = dataset.read(masked=True)

    result = ndxi_image(image, 'rb_ndvi')

    same = result == target

    assert same.all()


def test_ndsi(image):
    """
    Test ndvi
    """
    with rasterio.open('test/data/target/ndsi.tif') as dataset:
        target = dataset.read(masked=True)

    result = ndxi_image(image, 'ndsi')

    same = result == target

    assert same.all()


def test_hog():
    """
    Test hog
    """
    with Dataset("test/data/target/hog.nc", "r", format="NETCDF4") as dataset:
        target = dataset.variables['hog'][:]
        slices = dataset.variables['window'][:]

    window = slice(*slices[0:3]), slice(*slices[3:6])

    with rasterio.open(
            'test/data/baseimage/section_2_sentinel_grayscale.tif') as file:
        source = file.read(1, masked=True)

    features = hog_features(source[window])

    assert np.allclose(features, target)


def test_lacunarity():
    """
    Test lacunarity
    """
    with Dataset(
            "test/data/target/lacunarity.nc", "r",
            format="NETCDF4") as dataset:
        target = dataset.variables['lacunarity'][:]
        slices = dataset.variables['window'][:]

    window = slice(*slices[0:3]), slice(*slices[3:6])

    with rasterio.open(
            'test/data/baseimage/section_2_sentinel_canny_edge.tif') as file:
        source = file.read(1, masked=True).astype('bool')

    win = source[window]
    box_sizes = (10, 20)
    features = lacunarities(win, box_sizes)

    same = target == features

    assert same.all()


def test_pantex():
    """
    Test Pantex Feature
    """
    with Dataset(
            "test/data/target/pantex.nc", "r", format="NETCDF4") as dataset:
        target = dataset.variables['pantex'][:]
        slices = dataset.variables['window'][:]

    window = slice(*slices[0:3]), slice(*slices[3:6])

    with rasterio.open(
            'test/data/baseimage/section_2_sentinel_gray_ubyte.tif') as file:
        source = file.read(1, masked=True)

    features = [pantex(source[window])]

    same = target == features

    assert same.all()


def test_sift(image):
    """Sift feature test."""
    with Dataset("test/data/target/sift.nc", "r", format="NETCDF4") as dataset:
        target = dataset.variables['sift'][:]
        slices = dataset.variables['window'][:]

    window = slice(*slices[0:3]), slice(*slices[3:6])

    clusters = sift_cluster([image, image])

    win = image['gray_ubyte'][window]
    features = sift(win, clusters)

    same = target == features

    assert same.all()


def test_texton(image):
    """Texton feature test."""
    with Dataset(
            "test/data/target/texton.nc", "r", format="NETCDF4") as dataset:
        target = dataset.variables['texton'][:]
        slices = dataset.variables['window'][:]

    window = slice(*slices[0:3]), slice(*slices[3:6])

    clusters = texton_cluster([image, image], max_samples=1000)
    descriptors = get_texton_descriptors(image)

    win = descriptors[window]
    features = texton(win, clusters)

    same = target == features

    assert same.all()


@pytest.mark.parametrize('cluster_function', [sift_cluster, texton_cluster])
def test_cluster_max_samples(image, monkeypatch, cluster_function):
    max_samples = 1000

    def mock_fit(self, descriptors):
        samples = len(descriptors)
        assert 0 < samples <= max_samples

    monkeypatch.setattr(MiniBatchKMeans, 'fit', mock_fit)
    cluster_function([image, image, image],
                     max_samples=max_samples,
                     sample_window=(100, 100))
