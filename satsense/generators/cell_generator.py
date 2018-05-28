import math

from ..image import Image, SatelliteImage, Window


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
            x_pad_before = math.floor(math.fabs(x_start))
            x_start = 0
        if x_end > self.image.shape[0]:
            pad_needed = True
            x_pad_after = math.ceil(x_end - self.image.shape[0] + 1)
            x_end = self.image.shape[0] - 1
        if y_start < 0:
            pad_needed = True
            y_pad_before = math.floor(math.fabs(y_start))
            y_start = 0
        if y_end > self.image.shape[1]:
            pad_needed = True
            y_pad_after = math.ceil(y_end - self.image.shape[1] + 1)
            y_end = self.image.shape[1] - 1

        x_range = slice(x_start, x_end, 1)
        y_range = slice(y_start, y_end, 1)

        im = self.image.shallow_copy_range(x_range, y_range)
        if padding and pad_needed:
            im.pad(x_pad_before, x_pad_after, y_pad_before, y_pad_after)

        return Cell(im, self.x, self.y, x_range, y_range, orig=self.image)


class CellGenerator:
    def __init__(self, image: SatelliteImage, size: tuple, length=None):
        self.image = image

        self.x_size, self.y_size = size
        self.x_length = math.ceil(image.shape[0] / self.x_size)
        self.y_length = math.ceil(image.shape[1] / self.y_size)

        if length and length[0] < self.x_length:
            self.x_length = length[0]

        if length and length[1] < self.y_length:
            self.y_length = length[1]

    def __len__(self):
        return self.x_length * self.y_length

    @property
    def shape(self):
        return (self.x_length, self.y_length)

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

        x_start = x * self.x_size
        x_range = slice(x_start, x_start + self.x_size)

        y_start = y * self.y_size
        y_range = slice(y_start, y_start + self.y_size)

        im = self.image.shallow_copy_range(x_range, y_range)
        return Cell(im, x, y, x_range, y_range, orig=self.image)
