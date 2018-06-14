from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from six import iteritems


class Feature(object):
    """
    Feature superclass
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self._indices = None
        self._length = 0

    def __str__(self):
        return self.__class__.__name__

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


class FeatureSet(object):
    def __init__(self):
        self._features = OrderedDict()
        self._cur_index = 0

    def __iter__(self):
        return iter(self._features)

    def __str__(self):
        return self.string_presentation()

    def string_presentation(self):
        return "_".join([str(f) for k, f in self._features.items()])

    @property
    def items(self):
        return self._features

    def add(self, feature, name=None):
        if not name:
            name = "{0}-{1}".format(feature.__class__.__name__.upper(),
                                    len(self._features) + 1)

        self._features[name] = (feature)
        self._recalculate_feature_indices()

        return name, feature

    def remove(self, name):
        if name in self._features:
            del self._features[name]
            self._recalculate_feature_indices()
            return True
        return False

    @property
    def index_size(self):
        return self._cur_index

    def _recalculate_feature_indices(self):
        self._cur_index = 0
        for name, feature in iteritems(self._features):
            feature.indices = slice(self._cur_index,
                                    self._cur_index + feature.feature_size, 1)
            self._cur_index += feature.feature_size
