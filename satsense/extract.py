import numpy as np
from netCDF4 import Dataset


def extract_features(features, generator):
    """Compute features."""
    shape = generator.shape

    total_length = features.index_size
    print("Total length found: ", total_length)
    feature_vector = np.zeros((shape[0], shape[1], total_length))
    print("Feature vector:")
    print(feature_vector.shape)

    for cell in generator:
        for feature in features.items.values():
            feature_vector[cell.x, cell.y, feature.indices] = feature(cell)

    return feature_vector


def save_features(features, feature_vector, filename_prefix=''):
    """Save computed features."""
    for name, feature in features.items.items():
        filename = filename_prefix + name + '.nc'
        data = feature_vector[:, :, feature.indices]
        with Dataset(filename, 'w') as dataset:
            size_y, size_x, size_feature = data.shape
            dataset.createDimension('y', size_y)
            dataset.createDimension('x', size_x)
            dataset.createDimension('feature', size_feature)
            variable = dataset.createVariable(
                name, 'f4', dimensions=('y', 'x', 'feature'))
            variable[:] = data
