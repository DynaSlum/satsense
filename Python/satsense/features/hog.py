from satsense.util import get_rgb_image
from satsense.util.bands import RGB

from skimage.feature import hog as skihog
from skimage import data, color, exposure

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def hog(image, bands=RGB, normalize=True, orientations=9,
        pixels_per_cell=(8, 8), cells_per_block=(3, 3), visualise=True,
        transform_sqrt=False, feature_vector=True, block_norm='L2-Hys'):
    rgb_image = get_rgb_image(image, bands, normalize=normalize,)
    gray_image = color.rgb2gray(rgb_image)

    return skihog(gray_image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                  cells_per_block=cells_per_block, visualise=visualise,
                  transform_sqrt=transform_sqrt, block_norm=block_norm,
                  feature_vector=feature_vector)
