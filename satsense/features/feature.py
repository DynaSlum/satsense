from abc import ABC, abstractmethod


class Feature(ABC):
    """
    Feature superclass.

    Parameters
    ----------
        window_shapes : list[tuple]
            List of tuples of window shapes to calculate the feature on
        **kwargs : dict
            Keyword arguments for the feature

    Attributes
    ----------
    base_image
    """

    base_image = None
    """
    The base image this feature is calculated on
    ``Must be set by implementing classes``
    """

    size = None
    """
    The size of the feature in array shape
    ``Must be set by implementing classes``
    """

    def __init__(self, window_shapes, **kwargs):
        self._indices = {}
        self._length = 0
        self._windows = tuple(sorted(window_shapes, reverse=True))
        self.kwargs = kwargs
        self.name = self.__class__.__name__

    def __call__(self, window):
        return self.compute(window, **self.kwargs)

    @staticmethod
    @abstractmethod
    def compute(window, **kwargs):
        """
        Compute the feature on the window
        This function needs to be set by the implementation subclass
        ``compute = staticmethod(my_feature_calculation)``
        Parameters
        ----------
        window : tuple[int]
            The shape of the window
        **kwargs: dict
            The keyword arguments for the compustation
        """
        pass

    @property
    def windows(self):
        """
        Returns the windows this feature uses for calculation
        Returns
        -------
            tuple[tuple[int]]
        """
        return self._windows

    @windows.setter
    def windows(self, value):
        self._windows = tuple(sorted(value, reverse=True))

    @property
    def indices(self):
        """
        The indices for this feature in a feature set
        See Also
        --------
        FeatureSet
        """
        return self._indices

    @indices.setter
    def indices(self, value):
        self._indices = value


class FeatureSet():
    """
    FeatureSet Class

    The FeatureSet class can be used to bundle a number of features together.
    this class then calculates the indices for each feature within a vector
    of all features stacked into a single 3 dimensional matrix.
    """

    def __init__(self):
        self._features = {}
        self._cur_index = 0

    def __iter__(self):
        return iter(self._features)

    @property
    def items(self):
        return self._features.items()

    def add(self, feature, name=None):
        """
        Parameters
        ----------
        feature : Feature
            The feature to add to the set
        name : str
            The name to give the feature in the set.
            If none the features class name and length is used

        Returns:
            name : str
                The name of the added feature
            feature : Feature
                The added feature
        """
        if not name:
            name = "{0}-{1}".format(feature.__class__.__name__,
                                    len(self._features) + 1)
        feature.name = name
        self._features[name] = (feature)
        self._recalculate_feature_indices()

        return name, feature

    def remove(self, name):
        """
        Remove the feature from the set
        Parameters
        ----------
        name : str
            The name of the feature to remove

        Returns
        -------
        bool
            Wether the feature was succesfully removed
        """
        if name in self._features:
            del self._features[name]
            self._recalculate_feature_indices()
            return True
        return False

    @property
    def index_size(self):
        """
        The size of the index
        """
        return self._cur_index

    def _recalculate_feature_indices(self):
        self._cur_index = 0
        for feature in self._features.values():
            size = feature.size * len(feature.windows)
            feature.indices = slice(self._cur_index, self._cur_index + size)
            self._cur_index += size

    @property
    def base_images(self):
        """
        list[str]
            List of base images that was used to calculate these features
        """
        return {f.base_image for f in self._features.values()}
