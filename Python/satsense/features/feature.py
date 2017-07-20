from abc import ABCMeta, abstractmethod


class Feature:
    """
    Feature superclass
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self._indices = None
        self._length = 0

    @abstractmethod
    def __call__(self):
        pass

    @property
    def feature_size(self):
        return self._length

    @feature_size.setter
    def feature_size(self, value):
        self._length = value

    @property
    def indices(self):
        return self._indices

    @indices.setter
    def indices(self, value):
        self._indices = value
