from abc import ABC, abstractmethod


class Feature(ABC):
    """Feature superclass."""

    base_image = None
    size = None

    def __init__(self, window_shapes, **kwargs):
        self._indices = {}
        self._length = 0
        self._windows = window_shapes
        self.kwargs = kwargs
        self.name = self.__class__.__name__

    def __call__(self, window):
        return self.compute(window, **self.kwargs)

    @staticmethod
    @abstractmethod
    def compute(window, **kwargs):
        pass

    @property
    def windows(self):
        return self._windows

    @windows.setter
    def windows(self, value):
        self._windows = tuple(sorted(value, reverse=True))

    @property
    def indices(self):
        return self._indices

    @indices.setter
    def indices(self, value):
        self._indices = value


class FeatureSet():
    def __init__(self):
        self._features = {}
        self._cur_index = 0

    def __iter__(self):
        return iter(self._features)

    @property
    def items(self):
        return self._features.items()

    def add(self, feature, name=None):
        if not name:
            name = "{0}-{1}".format(feature.__class__.__name__,
                                    len(self._features) + 1)
        feature.name = name
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
        for feature in self._features.values():
            size = feature.size * len(feature.windows)
            feature.indices = slice(self._cur_index, self._cur_index + size)
            self._cur_index += size

    @property
    def base_images(self):
        return {f.base_image for f in self._features.values()}
