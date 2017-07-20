from . import RGB
from .generators import CellGenerator
import numpy as np


def calculate_feature_indices(features):
    cur_index = 0
    for feature in features:
        feature.indices = slice(cur_index, cur_index + feature.feature_size, 1)
        cur_index += feature.feature_size
    return cur_index


def extract_features(image, features, bands, cell_size=25, x_length=None, y_length=None):
    generator = CellGenerator(image, cell_size, x_length=x_length, y_length=y_length)
    shape = generator.shape()

    total_length = calculate_feature_indices(features)
    print("Total length found: ", total_length)
    feature_vector = np.zeros((shape[0], shape[1], total_length))
    print("Feature vector:")
    print(feature_vector.shape)
    for cell in generator:
        for feature in features:
            feature_vector[cell.x, cell.y, feature.indices] = feature(image, cell, bands=bands)

    return feature_vector
