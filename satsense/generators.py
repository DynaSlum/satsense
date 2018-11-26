"""Module providing a generator to iterate over the image."""
import logging
import math

from .image import Image

logger = logging.getLogger(__name__)


class BalancedGenerator():
    def __init__(self, image: Image, mask, ratio=1.0):
        """Balanced window generator.

        Select a balanced set of masked and non-masked point from the
        image and generators windows at those locations.
        """


class FullGenerator():
    def __init__(self,
                 image: Image,
                 step_size: tuple,
                 offset=(0, 0),
                 shape=None):
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

        block = []
        for i in range(2):
            offset = math.floor((self.offset[i] + 0.5) * self.step_size[i])
            start = offset - self._padding[i]
            end = (
                offset + self._padding[i] + self.shape[i] * self.step_size[i])
            block.append((start, end))
        block = tuple(block)
        image = self.image.copy_block(block)
        self._image_cache = image[itype]
        self.loaded_itype = itype

    def __iter__(self):
        if self._image_cache is None:
            raise RuntimeError("Please load an image first using load_image.")
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for window in self._windows:
                    yield self[i, j, window]

    def __getitem__(self, index):

        window = index[2]

        slices = []
        for i in range(2):
            middle = self._padding[i] + math.floor(
                (index[i] + 0.5) * self.step_size[i])
            start = math.floor(middle - 0.5 * window[i])
            end = start + window[i]
            slices.append(slice(start, end))

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
                shape=(row_length, self.shape[1]))
