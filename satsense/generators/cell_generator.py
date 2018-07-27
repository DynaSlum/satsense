"""Module providing a generator to iterate over the image."""
import logging
import math

from ..image import Image, SatelliteImage, Window

logger = logging.getLogger(__name__)


class Cell(Window):
    def __init__(self, image: Image, x, y, x_range, y_range, orig=None):
        super(Cell, self).__init__(image, x, y, x_range, y_range, orig=orig)

    def super_cell(self, size, padding=True):
        """
        """
        x_offset = (size[0] / 2.0)
        y_offset = (size[1] / 2.0)

        x_middle = (self.x_range.stop + self.x_range.start) / 2.0
        y_middle = (self.y_range.stop + self.y_range.start) / 2.0

        x_start = math.floor(x_middle - x_offset)
        x_end = math.floor(x_middle + x_offset)

        y_start = math.floor(y_middle - y_offset)
        y_end = math.floor(y_middle + y_offset)

        y_pad_before = 0
        y_pad_after = 0
        x_pad_before = 0
        x_pad_after = 0
        pad_needed = False
        if x_start < 0:
            pad_needed = True
            x_pad_before = -x_start
            x_start = 0
        if x_end > self.image.shape[0]:
            pad_needed = True
            x_pad_after = x_end - self.image.shape[0]
            x_end = self.image.shape[0]
        if y_start < 0:
            pad_needed = True
            y_pad_before = -y_start
            y_start = 0
        if y_end > self.image.shape[1]:
            pad_needed = True
            y_pad_after = y_end - self.image.shape[1]
            y_end = self.image.shape[1]

        x_range = slice(x_start, x_end)
        y_range = slice(y_start, y_end)

        img = self.image.shallow_copy_range(x_range, y_range)
        if padding and pad_needed:
            img.pad(x_pad_before, x_pad_after, y_pad_before, y_pad_after)

        return Cell(img, self.x, self.y, x_range, y_range, orig=self.image)


class CellGenerator:
    def __init__(self,
                 image: SatelliteImage,
                 size: tuple,
                 offset=(None, None),
                 length=(None, None)):
        self.image = image

        self.x_size, self.y_size = size
        self.x_length = math.ceil(image.shape[0] / self.x_size)
        self.y_length = math.ceil(image.shape[1] / self.y_size)

        offset = list(offset)
        if offset[0] is None:
            offset[0] = 0
        if offset[0] < 0:
            offset[0] = self.x_length - offset[0]
        if offset[0] > self.x_length:
            raise IndexError("x offset {} larger than image size {}".format(
                offset[0], self.x_length))
        self.x_offset = offset[0]

        if offset[1] is None:
            offset[1] = 0
        if offset[1] < 0:
            offset[1] = self.y_length - offset[1]
        if offset[1] > self.y_length:
            raise IndexError("y offset {} larger than image size {}".format(
                offset[1], self.y_length))
        self.y_offset = offset[1]

        length = list(length)
        if length[0] is None:
            length[0] = self.x_length
        if length[0] < 0:
            length[0] = self.x_length - length[0]
        if length[0] > self.x_length:
            raise IndexError("x length {} larger than image size {}".format(
                length[0], self.x_length))
        self.x_length = length[0]

        if length[1] is None:
            length[1] = self.y_length
        if length[1] < 0:
            length[1] = self.y_length - length[1]
        if length[1] > self.y_length:
            raise IndexError("y length {} larger than image size {}".format(
                length[1], self.y_length))
        self.y_length = length[1]

    def __len__(self):
        return self.x_length * self.y_length

    @property
    def shape(self):
        return (self.x_length, self.y_length)

    def split(self, n_jobs: int, features):
        """Split the CellGenerator into at most n_jobs."""
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
                getattr(self.image, itype)
                selected_images.add(itype)
        logger.debug("Images selected for job %s", selected_images)

        # Split the generator in chunks
        chunk = math.ceil(self.x_length / n_jobs)
        buffer = math.ceil(
            max(w[0] for f in features.items.values()
                for w in f.windows) / self.x_size)
        logger.debug("chunk size %s, buffer size %s cells", chunk, buffer)

        n_jobs = math.ceil(self.x_length / chunk)
        logger.debug(
            "Expecting to create %s generators based on image size %s with "
            "cell size %s and generator size %s", n_jobs, self.image.shape[0],
            self.x_size, self.x_length)
        for job in range(n_jobs):
            # Prepare image
            job_start = job * chunk
            image_start = max(0, (job_start - buffer) * self.x_size)
            image_end = min(self.image.shape[0],
                            (job_start + chunk + buffer) * self.x_size)
            image = self.image.shallow_copy_range(
                x_range=slice(image_start, image_end),
                y_range=slice(None),
                pad=False)
            logger.debug(
                "job %s, image start %s, image end %s, image shape %s", job,
                image_start, image_end, image.shape)
            self.image.collapse(selected_images)

            # Create generator to iterate over image
            offset = min(job_start, buffer)
            length = min(self.x_length - job_start, chunk)
            logger.debug(
                "job %s, job start %s, generator offset %s, length %s", job,
                job_start, offset, length)
            generator = CellGenerator(
                image,
                size=(self.x_size, self.y_size),
                offset=(offset, None),
                length=(length, None))
            logger.debug("generator shape %s", generator.shape)

            yield generator

    def __iter__(self):
        for x in range(self.x_length):
            for y in range(self.y_length):
                yield self[x, y]

    def __getitem__(self, index):
        x, y = index
        x = self.x_length - x if x < 0 else x
        y = self.y_length - y if y < 0 else y

        if x >= self.x_length or y >= self.y_length:
            raise IndexError('{} out of range for image of shape {}'.format(
                index, self.shape))

        x += self.x_offset
        y += self.y_offset

        x_start = x * self.x_size
        x_range = slice(x_start, x_start + self.x_size)

        y_start = y * self.y_size
        y_range = slice(y_start, y_start + self.y_size)

        im = self.image.shallow_copy_range(x_range, y_range)
        return Cell(im, x, y, x_range, y_range, orig=self.image)
