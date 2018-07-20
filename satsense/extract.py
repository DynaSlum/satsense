import logging
import math
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from os import cpu_count

import numpy as np
from netCDF4 import Dataset

from .generators import CellGenerator

logger = logging.getLogger(__name__)


def _get_multiple(value, base):
    """Get the nearest multiple of `base` that is larger than `value`."""
    multiple = 0
    while value > multiple * base:
        multiple += 1
    return multiple


def extract_features_parallel(features,
                              image,
                              cell_size,
                              max_jobs=cpu_count() // 2):
    """Extract features in parallel."""
    logger.debug("Extracting features using at most %s processes", max_jobs)

    img_size = image.shape[0]
    logger.debug("Image cells %s", math.ceil(img_size / cell_size[0]))

    # Compute number of cells in a job
    job_cells = _get_multiple(img_size // max_jobs + 1, cell_size[0])
    logger.debug('Job size: %s cells', job_cells)

    # Compute number of cells needed as buffer
    buffer_size = max(w[0] for f in features.items.values() for w in f.windows)
    buffer_cells = _get_multiple(buffer_size, cell_size[0])
    logger.debug('Buffer size: %s cells', buffer_cells)

    # Select the smallest possible set of images to pickle
    itypes = {f.base_image for f in features.items.values()}
    selected_images = set()
    # Images are derived in the following order, select only one
    order = ('raw', 'normalized', 'rgb', 'grayscale', 'gray_ubyte')
    for itype in order:
        if itype in itypes:
            selected_images.add(itype)
            break
    # Add images types that must be computed on the entire image
    for itype in ('canny_edge', 'texton_descriptors'):
        if itype in itypes:
            getattr(image, itype)
            selected_images.add(itype)
    logger.debug("Images selected for job %s", selected_images)

    # Create jobs
    generators = []
    n_jobs = 0
    end = 0
    while end < img_size:
        start = max(0, n_jobs * (job_cells - buffer_cells) * cell_size[0])
        end = min(img_size,
                  (n_jobs + 1) * (job_cells + buffer_cells) * cell_size[0])
        logger.debug("Creating job for x_range %s, %s", start, end)
        image_chunk = image.shallow_copy_range(
            x_range=slice(start, end), y_range=slice(None), pad=False)
        image_chunk.collapse(selected_images)
        generator = CellGenerator(image_chunk, cell_size)
        generators.append(generator)
        n_jobs += 1

    logger.debug("Will use %s processes to extract features", n_jobs)

    with ProcessPoolExecutor() as executor:
        # Submit jobs
        extract = partial(extract_features, features)
        result = executor.map(extract, generators)

        # Compute and store results
        # TODO: change generator so the spurious cells are not computed
        slices = [slice(buffer_cells, -buffer_cells) for _ in range(n_jobs)]
        if n_jobs > 1:
            slices[0] = slice(None, -buffer_cells)
            slices[-1] = slice(buffer_cells, None)
        else:
            slices[0] = slice(None)

        feature_vector = np.vstack(
            chunk[s] for chunk, s in zip(result, slices))

    logger.debug("Done extracting features. Feature vector shape %s",
                 feature_vector.shape)
    return feature_vector


def extract_features(features, generator):
    """Compute features."""
    shape = (generator.shape[0], generator.shape[1], features.index_size)
    feature_vector = np.zeros(shape, dtype=np.float32)
    logger.debug("Feature vector shape %s", shape)

    # Pre compute images
    itypes = {f.base_image for f in features.items.values()}
    logger.debug("Using base images: %s", ', '.join(itypes))
    for itype in itypes:
        getattr(generator.image, itype)

    size = len(generator)
    for i, cell in enumerate(generator):
        if i % (size // 10 or 1) == 0:
            logger.debug("%s%% ready", 100 * i // size)
        for feature in features.items.values():
            feature_vector[cell.x, cell.y, feature.indices] = feature(cell)

    return feature_vector


def save_features(features, feature_vector, filename_prefix=''):
    """Save computed features."""
    for name, feature in features.items.items():
        filename = filename_prefix + name + '.nc'
        logger.debug("Saving feature %s to file %s", name, filename)
        data = feature_vector[:, :, feature.indices]
        with Dataset(filename, 'w') as dataset:
            size_y, size_x, size_feature = data.shape
            dataset.createDimension('y', size_y)
            dataset.createDimension('x', size_x)
            dataset.createDimension('feature', size_feature)
            variable = dataset.createVariable(
                name, 'f4', dimensions=('y', 'x', 'feature'))
            variable[:] = data
