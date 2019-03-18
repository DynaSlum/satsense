"""Module providing a generator to iterate over the image."""
import logging
import math

import numpy as np

from .image import Image

logger = logging.getLogger(__name__)


class BalancedGenerator():
    """Balanced window generator."""

    def __init__(self,
                 image: Image,
                 masks,
                 p=None,
                 samples=None,
                 offset=(0, 0),
                 shape=None):
        """
        Constructor for BalancedGenerator class.

        Parameters
        ----------
        image: Image
            Satellite image
        masks: 1-D array-like
            List of masks, one for each class, to use for generating patches
            A mask should have a positive value for the array positions that
            are included in the class
        p: 1-D array-like, optional
            The probabilities associated with each entry in masks.
            If not given the sample assumes a uniform distribution
            over all entries in a.
        samples: int, optional
            The maximum number of samples to generate, otherwise infinite
        offset: tuple(int, int), optional
            Offset from the (0, 0) point (in number of steps)
        shape: tuple(int, int), optional
            Shape of the generator (in number of steps)

        example:
        BalancedGenerator(image,
                                 [
                                     class1_mask,
                                     class2_mask
                                     class3_mask
                                 ],
                                 [0.33, 0.33, 0.33])
        """
        self.image = image
        self.masks = masks
        self.p = p
        self.n = 0
        self.samples = samples

        if not shape:
            shape = image.shape
        self.shape = shape
        self.offset = offset

        # set using load_image
        self.loaded_itype = None
        self._image_cache = None
        self._windows = None
        self._padding = None

        if self.samples is None:
            self.samples = np.inf

        if p is not None and len(masks) is not len(p):
            raise ValueError(
                "The length of ratios needs to be the same as the number"
                "of masks, but they were masks: " + len(masks) +
                " vs ratios: " + len(p))

        if p is not None and not np.isclose(np.sum(self.p), 1.0):
            raise ValueError(
                "Ratio's should add up to (close to) 1.0, but added up to: " +
                str(np.sum(self.p)))

        self.crs = image.crs
        self.transform = image.transform

    def __iter__(self):
        """TODO#docstring."""
        if self._image_cache is None:
            raise RuntimeError("Please load an image first using load_image.")
        while self.n < self.samples:
            # Pick a random mask based on the probablities
            mask = np.random.choice(np.arange(len(self.masks)), p=self.p)

            # Pick a random location from the positive values of the mask
            valid = np.transpose(np.nonzero(self.masks[mask]))
            choice = np.random.choice(len(valid))
            self.n += 1

            for window in self._windows:
                yield self[valid[choice][0], valid[choice][1], window], mask

    def __getitem__(self, index):
        """
        TODO#docstring.

        Parameters
        ----------
            index:  1-D array-like
                TODO#docstring
        """
        window = index[2]

        slices = []
        for i in range(2):
            middle = self._padding[i] + index[i]
            start = math.floor(middle - 0.5 * window[i])
            end = start + window[i]
            slices.append(slice(start, end))

        return self._image_cache[slices[0], slices[1]]

    def load_image(self, itype, windows):
        """Load image with sufficient additional data to cover windows."""
        self._windows = tuple(sorted(windows, reverse=True))
        self._padding = tuple(
            max(math.ceil(0.5 * w[i]) for w in windows) for i in range(2))

        block = []
        for i in range(2):
            start = -self._padding[i]
            end = self.image.shape[i] + self._padding[i]
            block.append((start, end))
        block = tuple(block)
        image = self.image.copy_block(block)
        self._image_cache = image[itype]
        self.loaded_itype = itype

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
            row_offset = self.offset[0] + job * chunk_size[0]
            # col_offset = self.offset[1] + job * chunk_size[1]
            row_length = min(chunk_size[0], self.shape[0] - row_offset)
            # col_length = min(chunk_size[1], self.shape[1] - col_offset)
            if row_length <= 0:
                break
            # if col_length <= 0:
            #    break

            yield BalancedGenerator(
                image=self.image,
                masks=self.masks,
                p=self.p,
                samples=int(self.samples / n_chunks),
                offset=(row_offset, self.offset[1]),
                shape=(row_length, self.shape[1]))


class BalancedPatchGenerator():
    """Balanced patch generator."""

    def __init__(self,
                 image: Image,
                 mask,
                 samples=None,
                 offset=(0, 0),
                 shape=None):
        """
        Constructor for BalancedPatchGenerator.

        Parameters
        ----------
            image: Image
                Satellite image
            mask: 2-D array-like
                Image mask
            samples: int, optional
                Number of random samples to take
            offset: tuple(int, int), optional
                Offset from the (0, 0) point (in number of steps)
            shape: tuple(int, int), optional
                Shape of the generator (in number of steps)
        """
        self.image = image
        self.mask = mask
        self.n = 0
        self.samples = samples

        if not shape:
            shape = image.shape
        self.shape = shape
        self.offset = offset

        # set using load_image
        self.loaded_itype = None
        self._image_cache = None
        self._windows = None
        self._padding = None

        if self.samples is None:
            self.samples = np.inf

        self.crs = image.crs
        self.transform = image.transform

    def __iter__(self):
        """TODO#docstring."""
        if self._image_cache is None:
            raise RuntimeError("Please load an image first using load_image.")
        while self.n < self.samples:
            # Pick a random location from the positive values of the mask
            valid = np.transpose(np.nonzero(self.mask))
            choice = np.random.choice(len(valid))
            self.n += 1

            for window in self._windows:
                yield (self[valid[choice][0], valid[choice][1], window],
                       self.get_mask((valid[choice][0], valid[choice][1],
                                      window)))

    def load_image(self, itype, windows):
        """
        Load image with sufficient additional data to cover windows.

        Parameters
        ----------
            itype: str
                Image type
            windows: TODO#docstring
                TODO#docstring
        """
        self._windows = tuple(sorted(windows, reverse=True))
        self._padding = tuple(
            max(math.ceil(0.5 * w[i]) for w in windows) for i in range(2))

        block = []
        for i in range(2):
            start = -self._padding[i]
            end = self.image.shape[i] + self._padding[i]
            block.append((start, end))
        block = tuple(block)
        image = self.image.copy_block(block)
        self._image_cache = image[itype]
        self.loaded_itype = itype

    def __getitem__(self, index):
        """
        Extract item from image.

        Parameters
        ----------
            index: 1-D array-like
                TODO#docstring
        """
        window = index[2]

        slices = []
        for i in range(2):
            middle = self._padding[i] + index[i]
            start = math.floor(middle - 0.5 * window[i])
            end = start + window[i]
            slices.append(slice(start, end))

        return self._image_cache[slices[0], slices[1]]

    def get_mask(self, index):
        """
        Find if item at index is masked.

        Parameters
        ----------
            index: 1-D array-like
                TODO#docstring
        """
        window = index[2]

        slices = []
        for i in range(2):
            middle = index[i]
            start = math.floor(middle - 0.5 * window[i])
            end = start + window[i]
            slices.append(slice(start, end))

        return self.mask[slices[0], slices[1]]


class FullGenerator():
    """Window generator that covers the full image."""

    def __init__(self,
                 image: Image,
                 step_size: tuple,
                 offset=(0, 0),
                 shape=None,
                 with_slices=False):
        """Constructor for FullGenerator class.

        Parameters
        ----------
        image: Image
            Satellite image
        step_size: tuple(int, int)
            Size of the steps to use to iterate over the image (in pixels)
        offset: tuple(int, int)
            Offset from the (0, 0) point (in number of steps)
        shape: tuple(int, int)
            Shape of the generator (in number of steps)
        with_slices: bool
            Boolean indicating if slices should be used
        """
        self.image = image

        self.step_size = step_size
        self.offset = offset

        if not shape:
            shape = tuple(
                math.ceil(image.shape[i] / step_size[i]) for i in range(2))
        self.shape = shape

        self.crs = image.crs
        self.transform = image.scaled_transform(step_size)

        self.with_slices = with_slices

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
            windows: list of tuples
                The list of tuples of window shapes that will be used
                with this generator
        """
        self._windows = tuple(sorted(windows, reverse=True))
        self._padding = tuple(
            max(math.ceil(0.5 * w[i]) for w in windows) for i in range(2))

        block = self.get_blocks()
        image = self.image.copy_block(block)
        self._image_cache = image[itype]
        self.loaded_itype = itype

    def get_blocks(self):
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

    def get_slices(self, index, window):
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
            tuple of tuples : The x-range and y-range slices for the index and
                              window both with and without the padding included
        """
        slices = []
        paddless = []

        for i in range(2):
            start = self._padding[i] + (index[i] * self.step_size[i])
            end = start + window[i]
            slices.append(slice(start, end))
            paddless.append(
                slice(start - self._padding[i], end - self._padding[i]))

        return slices, paddless

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

        slices, paddless = self.get_slices(index, window)

        if self.with_slices:
            return self._image_cache[slices[0], slices[1]], slices, paddless
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
                shape=(row_length, self.shape[1]),
                with_slices=self.with_slices)
