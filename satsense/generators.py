"""Module providing a generator to iterate over the image."""
import logging
import math

from .image import Image

logger = logging.getLogger(__name__)


class BalancedGenerator():
    def __init__(self,
                 image: Image,
                 masks,
                 p=None,
                 samples=None,
                 offset=(0, 0),
                 shape=None):
        """Balanced window generator.

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

        example:
        BalancedGenerator(image,
                                 [
                                     class1_mask,
                                     class2_mask
                                     class3_mask
                                 ],
                                 [0.33, 0.33, 0.33])
        """
        raise NotImplementedError


class FullGenerator():
    def __init__(self,
                 image: Image,
                 step_size: tuple,
                 offset=(0, 0),
                 shape=None,
                 with_slices=False):
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
        """Load image with sufficient additional data to cover windows."""
        self._windows = tuple(sorted(windows, reverse=True))
        self._padding = tuple(
            max(math.ceil(0.5 * w[i]) for w in windows) for i in range(2))

        block = self._get_blocks()
        image = self.image.copy_block(block)
        self._image_cache = image[itype]
        self.loaded_itype = itype

    def _get_blocks(self):
        block = []
        for i in range(2):
            offset = self.offset[i] * self.step_size[i]
            start = offset - self._padding[i]
            end = (offset + self._padding[i] +
                   (self.shape[i] * self.step_size[i]))
            block.append((start, end))

        return tuple(block)

    def _get_slices(self, index, window):
        slices = []
        paddless = []

        for i in range(2):
            mid = self._padding[i] + math.floor(
                (index[i] + .5) * self.step_size[i])
            start = mid - math.floor(.5 * window[i])
            end = mid + math.ceil(.5 * window[i])
            slices.append(slice(start, end))
            paddless.append(
                slice(start - self._padding[i], end - self._padding[i]))

        return slices, paddless

    def __iter__(self):
        if self._image_cache is None:
            raise RuntimeError("Please load an image first using load_image.")
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for window in self._windows:
                    yield self[i, j, window]

    def __getitem__(self, index):
        window = index[2]

        slices, paddless = self._get_slices(index, window)

        if self.with_slices:
            return self._image_cache[slices[0], slices[1]], slices, paddless
        return self._image_cache[slices[0], slices[1]]

    def split(self, n_chunks):
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
