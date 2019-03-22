"""Histogram of Gradients feature."""
import cv2
import numpy as np
import scipy.stats

from .feature import Feature


def heaved_central_shift_moment(histogram, order):
    """Calculate the heaved central shift moment.

    Implementation is based on:
    Kumar, S., & Hebert, M. (2003, June). Man-made structure detection in
    natural images using a causal multiscale random field. In Computer vision
    and pattern recognition, 2003. proceedings. 2003 ieee computer society
    conference on (Vol. 1, pp. I-I). IEEE.

    Parameters
    ----------
    histogram : numpy.ndarray
        The histogram to calculate the moments over.
    order : int
        The order of the moment to calculate, a number between [0, inf).

    Returns
    -------
    float
        The heaved central shift moment.

    """
    if len(histogram.shape) > 1:
        raise ValueError("Can only calculate moments on a 1d array histogram, "
                         "but shape is: {}".format(histogram.shape))

    if order < 0:
        raise ValueError("Order cannot be below 0")

    mean = np.mean(histogram)

    # Moment 0 is just the mean
    if order == 0:
        return mean

    # In the paper they say: sum over all bins
    # The difference of the bin with the mean of the histogram (v0)
    # and multiply with a step function which is 1 when the difference is > 0
    diff = histogram - mean
    # The step function is thus a selection method, which is more easily
    # written like this.
    positive_diff = diff[diff > 0]

    power = order + 1
    numerator = np.sum(np.power(positive_diff, power))
    denominator = np.sum(positive_diff)

    if denominator == 0:
        moment = 0
    else:
        moment = numerator / denominator

    return moment


# @inproceedings('kumar2003man', {
#     'title': ('Man-made structure detection in natural images using a '
#               'causal multiscale random field'),
#     'author': 'Kumar, Sanjiv and Hebert, Martial',
#     'booktitle': ('Computer vision and pattern recognition, 2003. '
#                   'proceedings. 2003 ieee computer society conference on'),
#     'volume': '1',
#     'pages': 'I--I',
#     'year': '2003',
#     'organization': 'IEEE'
# })
def smoothe_histogram(histogram, kernel, bandwidth):
    """Vectorized histogram smoothing implementation.

    Implementation is based on:
    Kumar, S., & Hebert, M. (2003, June). Man-made structure detection in
    natural images using a causal multiscale random field. In Computer vision
    and pattern recognition, 2003. proceedings. 2003 ieee computer society
    conference on (Vol. 1, pp. I-I). IEEE.

    Parameters
    ----------
    histogram : numpy.ndarray
        The histogram to smoothe.
    kernel : function or callable object
        The kernel to use for the smoothing.
        For instance :obj:`scipy.stats.norm().pdf`.
    bandwidth : int
        The bandwidth of the smoothing.

    Returns
    -------
    numpy.ndarray
        The smoothed histogram.
    """
    if len(histogram.shape) > 1:
        raise ValueError("Can only smooth a 1d array histogram")

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


# My own binning algorithm. It does not do edge case detection because we're
# using the full 360 degrees the edge between 360 and 0 is there. I'm assuming
# no values over 360 exist.
def orientation_histogram(angles, magnitudes, number_of_orientations):
    """Create a histogram of orientations.

    Bins are created in the full 360 degrees.abs

    Parameters
    ----------
    angles : numpy.ndarray
        Angles of the orientations in degrees.
    magnitudes : numpy.ndarray
        Magnitude of the orientations.
    number_of_orientations: int
        The number of bins to use.

    Returns
    -------
    histogram: numpy.ndarray
        The histogram of orientations of shape number_of_orientations.
    bin_centers: numpy.ndarray
        The centers of the created bins with angles in degrees.

    """
    if len(angles.shape) > 2:
        raise ValueError("Only 2d windows are supported")

    if angles.shape != magnitudes.shape:
        raise ValueError(
            "Angle and magnitude arrays do not match shape: {} vs. {}".format(
                angles.shape, magnitudes.shape))

    number_of_orientations_per_360 = 360. / number_of_orientations

    histogram = np.zeros(number_of_orientations)
    bin_centers = np.zeros(number_of_orientations)

    for i in range(number_of_orientations):
        orientation_end = number_of_orientations_per_360 * (i + 1)
        orientation_start = number_of_orientations_per_360 * i
        bin_centers[i] = (orientation_end + orientation_start) / 2.
        select = (orientation_start <= angles) & (angles < orientation_end)
        histogram[i] = np.sum(magnitudes[select])

    return histogram, bin_centers


def hog_features(window, bins=50, kernel=None, bandwidth=0.7):
    """Calculate the hog features on the window.

    Features are the 1st and 2nd order heaved central shift moments,
    the angle of the two highest peaks in the histogram,
    the absolute sine difference between the two highest peaks.

    Parameters
    ----------
    window : numpy.ndarray
        The window to calculate the features on (grayscale).
    bands : dict
        A discription of the bands used in the window.
    bins : int
        The number of bins to use.
    kernel : :obj:`typing.Callable`
        The function to use for smoothing. The default is
        :obj:`scipy.stats.norm().pdf`.
    bandwidth: float
        The bandwidth for the smoothing.

    Returns
    -------
    :obj:`numpy.ndarray`
        The 5 HoG feature values.

    """
    if kernel is None:
        kernel = scipy.stats.norm().pdf

    mag, angle = cv2.cartToPolar(
        cv2.Sobel(window, cv2.CV_64F, 1, 0, ksize=3),
        cv2.Sobel(window, cv2.CV_64F, 0, 1, ksize=3),
        angleInDegrees=True,
    )

    histogram, bin_centers = orientation_histogram(angle, mag, bins)

    histogram = smoothe_histogram(histogram, kernel, bandwidth)

    # Calculate the Heaved Central-shift Moments
    # The first and second order are used as features
    moment1 = heaved_central_shift_moment(histogram, 1)
    moment2 = heaved_central_shift_moment(histogram, 2)

    # Find the two highest peaks. This is used for the following three features
    peaks = np.argsort(histogram)[::-1][0:2]

    # Feature 3 and 4: The absolute 'location' of the highest peak.
    # We can only interpret this as either the bin number, or the orientation
    # at the center of the bin.
    # That still doesn't give us 2 values, so we decided to take the center
    # orientations of the bins of the two highest peaks in degrees.

    delta1 = bin_centers[peaks[0]]
    delta2 = bin_centers[peaks[1]]

    # Feature 5: The absolute sine difference between the two highest peaks
    # Will be 1 when the two peaks are 90 degrees from eachother
    centers = np.deg2rad(bin_centers)
    beta = np.abs(np.sin(centers[peaks[0]] - centers[peaks[1]]))

    return np.array([moment1, moment2, delta1, delta2, beta])


class HistogramOfGradients(Feature):
    """
    Histogram of Oriented Gradient Feature Calculator

    The compute method calculates the feature on a particular
    window this returns the 1st and 2nd heaved central shift moments, the
    orientation of the first and second highest peaks and the absolute sine
    difference between the orientations of the highest peaks

    Parameters
    ----------
    window_shapes: list[tuple]
        The window shapes to calculate the feature on.
    bins : int
        The number of bins to use. The default is 50
    kernel : :obj:`typing.Callable`
        The function to use for smoothing. The default is
        :obj:`scipy.stats.norm().pdf`.
    bandwidth: float
        The bandwidth for the smoothing. The default is 0.7


    Attributes
    ----------
    size: int
        The size of the feature vector returned by this feature
    base_image: str
        The name of the base image used to calculate the feature


    Example
    -------
    Calculating the HistogramOfGradients on an image using a generator::

        from satsense import Image
        from satsense.generators import FullGenerator
        from satsense.extract import extract_feature
        from satsense.features import HistogramOfGradients

        windows = ((50, 50), )
        hog = HistogramOfGradients(windows)

        image = Image('test/data/source/section_2_sentinel.tif',
                      'quickbird')
        image.precompute_normalization()
        generator = FullGenerator(image, (10, 10))

        feature_vector = extract_feature(hog, generator)
    """
    base_image = 'grayscale'
    size = 5
    compute = staticmethod(hog_features)
