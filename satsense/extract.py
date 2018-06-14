import resource
import time
from functools import partial
from multiprocessing import Pool
from multiprocessing import cpu_count

import numpy as np
from numba import njit
from six import iteritems

from satsense.features import FeatureSet
from satsense.generators import CellGenerator
from satsense.util.cache import load_cache, cache


def timing(f):
    """
    Decorator to time function in minutes
    """

    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f minutes' % (f.__name__, (time2 - time1) / 60))
        return ret

    return wrap


@timing
def extract_features(features: FeatureSet, generator: CellGenerator, load_cached=True, image_name=""):
    start = time.time()
    shape = generator.shape()

    shared_feature_matrix = np.zeros((shape[0], shape[1], 1))
    print("\n--- Calculating Feature vector: {} ---\n".format(shared_feature_matrix.shape))

    for name, feature in iteritems(features.items):
        cache_key = "feature-{feature}-window{window}-image-{image_name}".format(
            image_name=image_name,
            window=(generator.x_size, generator.y_size),
            feature=str(feature),
        )

        feature_matrix = None
        if "Te-" not in str(feature):
            feature_matrix = load_cache(cache_key)

        if feature_matrix is None:
            feature_matrix = compute_feature(feature, generator)
            cache(feature_matrix, cache_key)

        if shared_feature_matrix:
            shared_feature_matrix = np.append(shared_feature_matrix, feature_matrix, axis=2)
        else:
            shared_feature_matrix = feature_matrix

            # Dirty fix. Would be better to re-use the windows every time so that
        # the windows do not have to be recalculated
        # (generator can only be iterated over once)
        generator = CellGenerator(generator.image, (generator.x_size, generator.y_size))

    end = time.time()
    print("Elapsed time extract multiprocessing: {} minutes, start: {}, end: {}".format((end - start) / 60, start, end))

    return shared_feature_matrix


@timing
def compute_feature(feature, generator):
    print("\n--- Calculating feature: {} ---\n".format(feature))

    start = time.time()

    shape = generator.shape()
    scales_feature_matrix = None
    # Calculate for different scales separately.
    for scale in feature.windows:
        # Prepare data for multiprocessing of individual features
        # Every feature needs 'initialize' option to work
        if hasattr(feature, 'initialize'):
            data = feature.initialize(generator, scale)
        else:
            raise ValueError("Initialize not implemented")

        end = time.time()
        print("Preparing data cells took {} seconds".format((end - start)))

        chunk_size = shape[0]
        cpu_cnt = cpu_count()

        # Get chunk size if feature has that function
        if hasattr(feature, 'chunk_size'):
            chunk_size = feature.chunk_size(cpu_cnt, shape)

        windows_chunked = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        total_chunks = len(windows_chunked)

        print("\nTotal chunks to compute: {}, chunk_size: {}".format(total_chunks, chunk_size))

        p = Pool(cpu_cnt, maxtasksperchild=1)
        compute_chunk_f = partial(compute_chunk, feature=feature)
        processing_results = p.map(compute_chunk_f, windows_chunked, chunksize=1)
        p.close()
        p.join()

        # Load individual results of processing back into one matrix
        feature_matrix = np.zeros((shape[0], shape[1], feature.feature_size))
        for coords, chunk_matrix in processing_results:
            load_results_into_matrix(feature_matrix, coords, chunk_matrix)

        if scales_feature_matrix:
            scales_feature_matrix = np.append(scales_feature_matrix, feature_matrix, axis=2)
        else:
            scales_feature_matrix = feature_matrix

        # Dirty fix. Would be better to re-use the windows every time so that
        # the windows do not have to be recalculated
        # (generator can only be iterated over once)
        generator = CellGenerator(generator.image, (generator.x_size, generator.y_size))

    return scales_feature_matrix


@njit
def load_results_into_matrix(feature_matrix, coords, chunk_matrix):
    for i in range(coords.shape[0]):
        x = int(coords[i, 0])
        y = int(coords[i, 1])
        fv = chunk_matrix[i, :]
        feature_matrix[x, y, :] = fv


def compute_chunk(chunk, feature):
    start = time.time()

    result = feature(chunk)

    end = time.time()
    delta = (end - start)
    per_block = delta / len(chunk)
    mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    print(
        "Calculating {} Windows took {} seconds,"
        " each window took: {} (avg), mem:{}".format(len(chunk), delta, per_block,
                                                     mem_usage))

    return result
