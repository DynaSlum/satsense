import numpy as np


class Dataset:
    def __init__(self, feature_vector):
        self._feature_vector = feature_vector

    def createXY(self, mask, remove_out=True, in_label=1, out_label=0):
        if mask.shape[:2] == self._feature_vector.shape[:2]:
            nrows = self._feature_vector.shape[0] * self._feature_vector.shape[1]
            nfeatures = self._feature_vector.shape[2]

            X = np.reshape(self._feature_vector, (nrows, nfeatures))
            y = np.reshape(np.copy(mask), (nrows, ))
            
            if remove_out:
                X = X[y == 1]
                y = y[y == 1]

            # copy y because labels might be swapped
            y_temp = np.copy(y)
            if in_label != 1:
                y[y_temp == 1] = in_label
            if out_label != 0:
                y[y_temp == 0] = out_label

            return X, y
        else:
            raise ValueError("Mask shape {}does not match feature vector shape{}".format(mask.shape[:2], self._feature_vector.shape[:2]))
