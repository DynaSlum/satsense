from satsense.util import RGB
from math import ceil, floor, fabs
from collections import namedtuple
import numpy as np


class CellGenerator:
    Cell = namedtuple('Cell', ['x', 'y', 'x_range', 'y_range', 'window'])

    def __init__(self, image, size, x_length=None, y_length=None):
        self.cur_x = 0
        self.cur_y = 0

        self.size = size
        self.image = image

        if x_length:
            self.x_length = x_length
        else:
            self.x_length = ceil(image.shape[0] / size)

        if y_length:
            self.y_length = y_length
        else:
            self.y_length = ceil(image.shape[1] / size)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def len(self):
        return self.x_length * self.y_length

    def shape(self):
        return self.x_length, self.y_length

    def next(self):
        try:
            x, y, x_range, y_range = self.next_slice()
            window = self.image[x_range, y_range]
            return CellGenerator.Cell(x, y, x_range, y_range, window)
        except StopIteration as e:
            raise e
        # return {
        #     'x': x,
        #     'y': y,
        #     'x_range': x_range,
        #     'y_range': y_range,
        #     'window': self.image[x_range, y_range]
        # }

    def next_slice(self):
        if self.cur_y < self.y_length:
            x_start = self.size * self.cur_x
            x_end = self.size * (self.cur_x+1)
            y_start = self.size * self.cur_y
            y_end = self.size * (self.cur_y+1)
            self.cur_y += 1

            return self.cur_x, self.cur_y - 1, slice(x_start, x_end, 1), slice(y_start, y_end, 1)
        elif self.cur_x + 1 < self.x_length:
            self.cur_y = 0
            self.cur_x += 1

            x_start = self.size * self.cur_x
            x_end = self.size * (self.cur_x+1)
            y_start = self.size * self.cur_y
            y_end = self.size * (self.cur_y+1)
            return self.cur_x, self.cur_y, slice(x_start, x_end, 1), slice(y_start, y_end, 1)
        else:
            raise StopIteration()

def SuperCell(image, cell, size, padding=True):
    """
    """
    x_range = cell.x_range # From CellGenerator
    y_range = cell.y_range

    offset = (size / 2.0)

    x_middle = (x_range.stop + x_range.start) / 2.0
    y_middle = (y_range.stop + y_range.start) / 2.0

    x_start = floor(x_middle - offset)
    x_end = floor(x_middle + offset)

    y_start = floor(y_middle - offset)
    y_end = floor(y_middle + offset)

    y_pad_before = 0
    y_pad_after = 0
    x_pad_before = 0
    x_pad_after = 0
    pad_needed = False
    if x_start < 0:
        pad_needed = True
        x_pad_before = floor(fabs(x_start))
        x_start = 0
    if x_end > image.shape[0]:
        pad_needed = True
        x_pad_after = ceil(x_end - image.shape[0] + 1)
        x_end = image.shape[0] - 1
    if y_start < 0:
        pad_needed = True
        y_pad_before = floor(fabs(y_start))
        y_start = 0
    if y_end > image.shape[1]:
        pad_needed = True
        y_pad_after = ceil(y_end - image.shape[1] + 1)
        y_end = image.shape[1] - 1

    x_range = slice(x_start, x_end, 1)
    y_range = slice(y_start, y_end, 1)

    window = image[x_range, y_range]

    if padding and pad_needed:
        window = np.pad(window, ((x_pad_before, x_pad_after), (y_pad_before, y_pad_after), (0, 0)), 'constant', constant_values=0)

    return CellGenerator.Cell(cell[0], cell[1], x_range, y_range, window)


def calculate_feature_indices(features):
    cur_index = 0
    for feature in features:
        feature.indices = slice(cur_index, cur_index + feature.feature_size, 1)
        cur_index += feature.feature_size
    return cur_index


def extract_features(image, features, bands, cell_size=25, x_length=None, y_length=None):
    generator = CellGenerator(image, cell_size, x_length=x_length, y_length=y_length)
    shape = generator.shape()

    total_length = calculate_feature_indices(features)
    print("Total length found: ", total_length)
    feature_vector = np.zeros((shape[0], shape[1], total_length))
    print("Feature vector:")
    print(feature_vector.shape)
    for cell in generator:
        for feature in features:
            feature_vector[cell.x, cell.y, feature.indices] = feature(image, cell, bands=bands)

    return feature_vector
