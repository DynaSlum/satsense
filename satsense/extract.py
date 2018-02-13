from . import RGB
from .generators import CellGenerator
import numpy as np
from six import iteritems

def extract_features(features, generator):
    shape = generator.shape()

    total_length = features.index_size
    print("Total length found: ", total_length)
    feature_vector = np.zeros((shape[0], shape[1], total_length))
    print("Feature vector:")
    print(feature_vector.shape)
    for window in generator:
        for name, feature in iteritems(features.items):
            feature_vector[window.x, window.y, feature.indices] = feature(window)

    return feature_vector
