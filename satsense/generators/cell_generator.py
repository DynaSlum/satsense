import math
from collections import namedtuple
from ..image import Image, Window, SatelliteImage

import numpy as np

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
        self.cur_x = 0
        self.cur_y = -1

        self.x_size, self.y_size = size
        self.image = image

        self.x_length = math.ceil(image.shape[0] / self.x_size)
        self.y_length = math.ceil(image.shape[1] / self.y_size)

        if length and length[0] < self.x_length:
            self.x_length = length[0]
        
        if length and length[1] < self.y_length:
            self.y_length = length[1]


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def get(self, index):
        return self.cell_from_slice(self.slice_at_index(index))

    def len(self):
        return self.x_length * self.y_length

    def shape(self):
        return self.x_length, self.y_length

    def next(self):
        return self.cell_from_slice(self.next_slice())

    def cell_from_slice(self, the_slice):
        x, y, x_range, y_range = the_slice
        im = self.image.shallow_copy_range(x_range, y_range)
        return Cell(im, x, y, x_range, y_range, orig=self.image)

    def next_slice(self):
        if self.cur_y + 1 < self.y_length:
            self.cur_y += 1
        elif self.cur_x + 1 < self.x_length:
            self.cur_y = 0
            self.cur_x += 1
        else:
            raise StopIteration

        x_start = self.cur_x * self.x_size
        x_end = x_start + self.x_size
        y_start = self.cur_y * self.y_size
        y_end = y_start + self.y_size
        return self.cur_x, self.cur_y, slice(x_start, x_end), slice(y_start, y_end)

    def slice_at_index(self, index):
        x = math.floor(index / self.y_length)
        y = index % self.y_length

        if x < self.x_length and y < self.y_length and x >= 0 and y >= 0:
            x_start = self.x_size * x
            x_end = self.x_size * (x+1)
            y_start = self.y_size * y
            y_end = self.y_size * (y+1)

            return x, y - 1, slice(x_start, x_end, 1), slice(y_start, y_end, 1)
        else:
            raise IndexError("index out of range:", index,
                             " e.g. (", x, ",", y, ") does not fall within",
                             "image bounds: (", self.x_size, ",", self.y_size, ")")
