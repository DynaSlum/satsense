import rasterio
import numpy as np

from satsense.features.hog import hog_features
from satsense.features.lacunarity import lacunarity
from netCDF4 import Dataset


def write_target(array, name, fname, w):
    dataset = Dataset(name, 'w', format="NETCDF4")

    dataset.createDimension('window_len', 6)
    variable = dataset.createVariable('window', 'i4',
                                      dimensions=('window_len'))
    variable[:] = [w[0].start, w[0].stop, w[0].step,
                   w[1].start, w[1].stop, w[1].step]

    dataset.createDimension('length', len(array))
    variable = dataset.createVariable(fname, 'f8', dimensions=('length'))
    variable[:] = array

    dataset.close()


def hog_target():
    with rasterio.open(
            '../baseimage/section_2_sentinel_grayscale.tif') as gray:
        source = gray.read(1, masked=True)

        window = (slice(100, 125, 1), slice(100, 125, 1))
        features = hog_features(source[window])

        write_target(features, '../target/hog.nc', 'hog', window)


def lacunarity_target():
    with rasterio.open(
            '../baseimage/section_2_sentinel_canny_edge.tif') as canny:
        source = canny.read(1, masked=True).astype('bool')

        window = (slice(100, 125, 1), slice(100, 125, 1))

        box_sizes = (10, 20)
        result = np.zeros(len(box_sizes))
        for i, box_size in enumerate(box_sizes):
            win = source[window].filled()
            result[i] = lacunarity(win, box_size)

        write_target(result, '../target/lacunarity.nc', 'lacunarity', window)


if __name__ == '__main__':
    hog_target()
    lacunarity_target()
