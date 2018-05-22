import numpy as np
from numba import jit


@jit(nopython=True)
def count_codewords(codewords, vector_length):
    histogram = np.zeros((vector_length), dtype=np.int32)
    for codeword in codewords:
        histogram[codeword] += 1

    return histogram
