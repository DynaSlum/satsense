import cv2
import numpy as np
import scipy.stats

from ..bands import RGB
from .feature import Feature


def heaved_central_shift_moment(histogram, order):
    """
    Calculate the heaved central shift moment of a histogram
    of the given order.

    Implementation is based on:
    Kumar, S., & Hebert, M. (2003, June). Man-made structure detection in natural
    images using a causal multiscale random field. In Computer vision and pattern
    recognition, 2003. proceedings. 2003 ieee computer society conference on
    (Vol. 1, pp. I-I). IEEE.

    Parameters
    ----------
    histogram : numpy.ndarray
        The histogram to calculate the moments over
    order : int
        The order of the moment to calculate, a number between [0, inf) 
    """
    if len(histogram.shape) > 1:
        raise (
            "Can only calculate moments on a 1d array histogram, but shape is:"
            + histogram.shape)

    if order < 0:
        raise ("Order cannot be below 0")

    bins = histogram.shape[0]
    v0 = np.mean(histogram)

    # Moment 0 is just the mean
    if order == 0:
        return v0

    total = 0
    # In the paper they say: sum over all bins
    # The difference of the bin with the mean of the histogram (v0)
    # and multiply with a step function which is 1 when the difference is > 0
    diff = histogram - v0
    # The step function is thus a selection method, which is more easily written like this
    positive_diff = diff[diff > 0]

    power = order + 1
    numerator = np.sum(np.power(positive_diff, power))
    denominator = np.sum(positive_diff)

    if denominator == 0:
        v = 0
    else:
        v = numerator / denominator

    return v


# @inproceedings('kumar2003man', {
#     'title': 'Man-made structure detection in natural images using a causal multiscale random field',
#     'author': 'Kumar, Sanjiv and Hebert, Martial',
#     'booktitle': 'Computer vision and pattern recognition, 2003. proceedings. 2003 ieee computer society conference on',
#     'volume': '1',
#     'pages': 'I--I',
#     'year': '2003',
#     'organization': 'IEEE'
# })
def smoothe_histogram(histogram, kernel, bandwidth):
    """
    Vectorized histogram smoothing implementation

    Implementation is based on:
    Kumar, S., & Hebert, M. (2003, June). Man-made structure detection in natural
    images using a causal multiscale random field. In Computer vision and pattern
    recognition, 2003. proceedings. 2003 ieee computer society conference on
    (Vol. 1, pp. I-I). IEEE.

    Parameters
    ----------
    histogram : numpy.ndarray
        The histogram to smoothe
    kernel : function or callable object
        The kernel to use for the smoothing. For instance scipy.stats.norm().pdf
    bandwidth : int
        The bandwidth of the smoothing.

    Returns
    -------
    The smoothed histogram
    """
    if len(histogram.shape) > 1:
        raise ("Can only smooth a 1d array histogram")

    bins = histogram.shape[0]

    # Make a bins x bins matrix with the inter-bin distances
    # Equivalent to:
    # for i in bins:
    #     for j in bins
    #         matrix[i, j] = (i - j) / bandwidth
    matrix = np.array([i - np.arange(bins)
                       for i in np.arange(bins)]) / bandwidth
    smoothing_matrix = kernel(matrix)

    smoothing_factor_totals = np.sum(smoothing_matrix, axis=1)
    pre_smooth_histogram = np.sum(smoothing_matrix * histogram, axis=1)

    smoothed_histogram = pre_smooth_histogram / smoothing_factor_totals

    return smoothed_histogram


# My own binning algorithm. It does not do edge case detection because we're using the full 360 degrees the
# edge between 360 and 0 is there. I'm assuming no values over 360 exist.
def orientation_histogram(angles, magnitudes, number_of_orientations):
    """
    Creates a histogram of orientations. Bins are created in the full 360 degrees.abs

    Parameters
    ----------
    angles : numpy.ndarray
        Angles of the orientations in degrees
    magnitudes : numpy.ndarray
        Magnitude of the orientations
    number_of_orientations: int
        The number of bins to use

    Returns
    -------
    histogram: numpy.ndarray
        The histogram of orientations of shape number_of_orientations
    bin_centers: numpy.ndarray
        The centers of the created bins with angles in degrees
    """
    if len(angles.shape) > 2:
        raise ("Only 2d windows are supported")

    if angles.shape != magnitudes.shape:
        raise ("Angle and magnitude arrays do not match shape: {0} vs. {1}".
               format(angles.shape, magnitudes.shape))

    number_of_orientations_per_360 = 360. / number_of_orientations
    x_size, y_size = angles.shape

    histogram = np.zeros(number_of_orientations)
    bin_centers = np.zeros(number_of_orientations)

    for i in range(number_of_orientations):
        orientation_end = number_of_orientations_per_360 * (i + 1)
        orientation_start = number_of_orientations_per_360 * i
        bin_centers[i] = (orientation_end + orientation_start) / 2.

        total = 0
        for x in range(x_size):
            for y in range(y_size):
                if angles[x,
                          y] >= orientation_start and angles[x,
                                                             y] < orientation_end:
                    total += magnitudes[x, y]

        histogram[i] = total
    return histogram, bin_centers


def hog_features(window,
                 bands=RGB,
                 bins=50,
                 kernel=scipy.stats.norm().pdf,
                 bandwidth=0.7):
    """
    Calculates the hog features on the window.
    Features are the 1st and 2nd order heaved central shift moments
    The angle of the two highest peaks in the histogram
    The absolute sine difference between the two highest peaks

    Parameters
    ----------
    window : numpy.ndarray
        The window to calculate the features on (grayscale)
    bands : dict
        A discription of the bands used in the window
    bins : number
        The number of bins to use
    kernel : function or callable object
        The function to use for smoothing
    bandwidth:
        The bandwidth for the smoothing
    """
    gx = cv2.Sobel(window, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(window, cv2.CV_64F, 0, 1, ksize=3)

    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    histogram, bin_centers = orientation_histogram(angle, mag, bins)

    smoothed_histogram = smoothe_histogram(histogram, kernel, bandwidth)

    # Calculate the Heaved Central-shift Moments
    # The first and second order are used as features
    v1 = heaved_central_shift_moment(smoothed_histogram, 1)
    v2 = heaved_central_shift_moment(smoothed_histogram, 2)

    # Find the two highest peaks. This is used for the following three features
    peaks = np.argsort(smoothed_histogram)[::-1][0:2]

    # Feature 3 and 4: The absolute 'location' of the highest peak.
    # We can only interpret this as either the bin number, or the orientation at the center of the bin
    # That still doesn't give us 2 values, so we decided to take the center orientations of the bins of the
    # two highest peaks in degrees

    delta1 = bin_centers[peaks[0]]
    delta2 = bin_centers[peaks[1]]

    # Feature 5: The absolute sine difference between the two highest peaks
    # Will be 1 when the two peaks are 90 degrees from eachother
    centers = np.deg2rad(bin_centers)
    beta = np.abs(np.sin(centers[0] - centers[1]))

    return np.array([v1, v2, delta1, delta2, beta])


class HistogramOfGradients(Feature):
    def __init__(self, windows=((25, 25), )):
        super(HistogramOfGradients, self)
        self.windows = windows
        self.feature_len = 5
        print(self.windows)
        self.feature_size = self.feature_len * len(self.windows)

    def __call__(self, cell):
        result = np.zeros(self.feature_size)
        for i, window in enumerate(self.windows):
            win = cell.super_cell(window, padding=True)

            result[i * self.feature_len:(i + 1) *
                   self.feature_len] = hog_features(
                       win.grayscale,
                       bands=win.bands,
                       bins=50,
                       kernel=scipy.stats.norm().pdf,
                       bandwidth=0.7)
        return result
