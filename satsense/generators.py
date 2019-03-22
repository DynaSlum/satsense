"""Module providing a generator to iterate over the image."""
import logging
import math

from .image import Image

logger = logging.getLogger(__name__)


class BalancedGenerator():
    """
    Balanced window generator.

    Parameters
    ----------
    image : Image
        Satellite image
    masks : 1-D array-like
        List of masks, one for each class, to use for generating patches
        A mask should have a positive value for the array positions that
        are included in the class
    p : 1-D array-like, optional
        The probabilities associated with each entry in masks.
        If not given the sample assumes a uniform distribution
        over all entries in a.
    samples : int, optional
        The maximum number of samples to generate, otherwise infinite

    Examples
    ---------
    Using BalancedGenerator

        >>> from satsense.generators import BalancedGenerator
        >>> BalancedGenerator(image,
                              [class1_mask, class2_mask, class3_mask],
                              [0.33, 0.33, 0.33])
    """

    def __init__(self,
                 image: Image,
                 masks,
                 p=None,
                 samples=None,
                 offset=(0, 0),
                 shape=None):
        raise NotImplementedError


class FullGenerator():
    """Window generator that covers the full image.

    Parameters
    ----------
    image: Image
        Satellite image
    step_size: tuple(int, int)
        Size of the steps to use to iterate over the image (in pixels)
    offset: tuple(int, int)
        Offset from the (0, 0) point (in number of steps).
    shape: tuple(int, int)
        Shape of the generator (in number of steps)

    """

    def __init__(self,
                 image: Image,
                 step_size: tuple,
                 offset=(0, 0),
                 shape=None):
        self.image = image

        self.step_size = step_size
        self.offset = offset

        if not shape:
            shape = tuple(
                math.ceil(image.shape[i] / step_size[i]) for i in range(2))
        self.shape = shape

        self.crs = image.crs
        self.transform = image.scaled_transform(step_size)

        # set using load_image
        self.loaded_itype = None
        self._image_cache = None
        self._windows = None
        self._padding = None

    def load_image(self, itype, windows):
        """
        Load image with sufficient additional data to cover windows.

        Parameters
        ----------
            itype: str
                Image type
            windows: list[tuple]
                The list of tuples of window shapes that will be used
                with this generator
        """
        self._windows = tuple(sorted(windows, reverse=True))
        self._padding = tuple(
            max(math.ceil(0.5 * w[i]) for w in windows) for i in range(2))

        block = self._get_blocks()
        image = self.image.copy_block(block)
        self._image_cache = image[itype]
        self.loaded_itype = itype

    def _get_blocks(self):
        """
        Calculate the size of the subset needed to include enough
        data for the calculations of windows for this generator
        """
        block = []
        for i in range(2):
            offset = self.offset[i] * self.step_size[i]
            start = offset - self._padding[i]
            end = (offset + self._padding[i] +
                   (self.shape[i] * self.step_size[i]))
            block.append((start, end))

        return tuple(block)

    def _get_slices(self, index, window):
        """
        Calculate the array slices needed to retrieve the window from the image
        at the provided index

        Parameters
        ----------
            index:  1-D array-like
                The x and y coordinates for the slice in steps
            window: 1-D array-like
                The x and y size of the window

        Returns
        -------
            tuple[tuple]
                The x-range and y-range slices for the index and
                window both with and without the padding included
        """
        slices = []

        for i in range(2):
            mid = self._padding[i] + math.floor(
                (index[i] + .5) * self.step_size[i])
            start = mid - math.floor(.5 * window[i])
            end = start + window[i]
            slices.append(slice(start, end))

        return slices

    def __iter__(self):
        """
        Iterate over the x and y coordinates of the generator and windows

        While iterating it will return for each x and y coordinate as defined
        by the step_size the part of the image as defined by the window.

        Consecutive calls will first return each window and then move to the
        next coordinates

        Returns
        -------
            collections.Iterable[numpy.ndarray]
        """
        if self._image_cache is None:
            raise RuntimeError("Please load an image first using load_image.")
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for window in self._windows:
                    yield self[i, j, window]

    def __getitem__(self, index):
        """
        Extract item from image.

        Parameters
        ----------
            index: 1-D array-like
                An array wich specifies the x and y coordinates
                and the window shape to get from the generator

        Examples:
        ---------
        >>> generator[0, 0, (100, 100)]
        """
        window = index[2]

        slices = self._get_slices(index, window)

        return self._image_cache[slices[0], slices[1]]

    def split(self, n_chunks):
        """
        Split processing into chunks.

        Parameters
        ----------
            n_chunks: int
                Number of chunks to split the image into
        """
        chunk_size = math.ceil(self.shape[0] / n_chunks)
        for job in range(n_chunks):
            row_offset = self.offset[0] + job * chunk_size
            row_length = min(chunk_size, self.shape[0] - row_offset)
            if row_length <= 0:
                break
            yield FullGenerator(
                image=self.image,
                step_size=self.step_size,
                offset=(row_offset, self.offset[1]),
                shape=(row_length, self.shape[1]))
