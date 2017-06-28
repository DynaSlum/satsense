from ..util import RGB, get_grayscale_image

from skimage.feature import hog as skihog

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def hog(window, bands=RGB, orientations=9,
        pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualise=True,
        transform_sqrt=False, feature_vector=True, block_norm='L2-Hys'):
    gray_image = get_grayscale_image(window, bands=bands)

    return skihog(gray_image, orientations=orientations,
                  pixels_per_cell=pixels_per_cell,
                  cells_per_block=cells_per_block, visualise=visualise,
                  transform_sqrt=transform_sqrt, block_norm=block_norm,
                  feature_vector=feature_vector)
