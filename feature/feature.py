from collections import UserList, UserDict
from functools import partial


def fnv32a(text):
    h = 0x811c9dc5
    for c in text:
        h = ((h ^ ord(c)) * 0x01000193) & 0xffffffff
    return h


def numbered_columns(array):
    # Can't use `not array` because numpy interprets it differently.
    if len(array) == 0:
        return []
    return [str(i) for i in range(len(array[0]))]


def iterate_items(obj):
    """Iterates over the object's key value pairs (key, value), where obj[key] == value."""
    if hasattr(obj, "items"):
        return obj.items()
    # If the object lacks the items() function, assume it's a list with keys [0, len(obj)-1].
    return enumerate(obj)


class Array(UserList):
    """Array stores the real-valued features.

    Args:
        columns: [str], the name of each column.
    """

    def __init__(self, columns=None):
        super().__init__()
        self._columns = list(columns) if columns is not None else []
        self.data = []

    def concat(self, other, prefix=""):
        """Adds the columns from `other` to `self`. `other` can be
        of any type that supports length, indexing and enumeration. Furthermore
        column names can be supplied in a `columns` attribute.

        Args:
            other: contains the new columns.
            prefix: str, optional prefix added to the new column names
                to avoid name clashes.
        """
        if len(self) == 0:
            self.data = [[] for i in range(len(other))]

        if len(self) != len(other):
            raise ValueError("array length does not match - have {} and {}".format(len(self), len(other)))

        columns = other.columns if hasattr(other, "columns") else numbered_columns(other)
        if prefix:
            columns = ["{}_{}".format(prefix, name) for name in columns]

        # Make sure the column names do not clash.
        for column in columns:
            if column in self._columns:
                raise ValueError("a column named '{}' already exists".format(column))

        self._columns += columns
        for i, row in enumerate(other):
            self.data[i].extend(row)

    @property
    def shape(self):
        """Returns the array shape."""
        return (len(self.data), len(self.data[0]))

    @property
    def columns(self):
        """Returns the column names."""
        if not self._columns:
            self._columns = numbered_columns(self)
        return self._columns


class BaseFeature(object):
    def set(self, *args, **kwargs):
        raise NotImplementedError()

    def push(self):
        raise NotImplementedError()

    def array(self):
        raise NotImplementedError()

class CurriedSet(object):
    """Helper class that allows partially applying arguments to the object's set function
    by putting them into the function name. This allows more readable function calls:

    >>> obj.set("a", "b", "c", 1.0)
    >>> obj.set_a_b_c(1.0)
    """

    def __getattr__(self, name):
        function, *partial_applied_args = name.split("_")
        if function != "set" or not partial_applied_args:
            return self.__getattribute__(name)
        return partial(self.set, *partial_applied_args)


class Pipe(object):
    """Pipe applies one ore more functions to a feature array.

    Args:
        feature: Feature|Group, of which the array shall be transformed.
        functions: callable, one ore more functions that take and return an array.
    """

    def __init__(self, feature, *functions):
        self.feature = feature
        self.functions = functions

    def array(self):
        """Returns the feature array with all functions of the pipe applied."""
        result = self.feature.array()
        for function in self.functions:
            result = function(result)
        return result

    def __getattr__(self, name):
        return getattr(self.feature, name)


class Group(BaseFeature, CurriedSet):
    """Group produces real-valued feature arrays from one or more
    Feature/Group classes.

    Args:
        features: {str: Feature|Group}, instances of Feature/Group classes
            stored under their names.
    """

    def __init__(self, features):
        super().__init__()
        self.features = features

    def set(self, *args, **kwargs):
        # If there's only a single feature, allow omitting the feature name.
        # Otherwise name should be the first argument and the remaining arguments
        # get passed on to the feature's set() function.
        if len(self.features) == 1:
            name, = self.features.keys()
            if name == args[0]:
                args = args[1:]
        else:
            name = args[0]
            args = args[1:]
        self.features[name].set(*args, **kwargs)

    def push(self):
        for feature in self.features.values():
            feature.push()

    def array(self):
        result = Array()
        for name, feature in sorted(self.features.items()):
            result.concat(feature.array(), prefix=name)
        return result


class Feature(BaseFeature):
    """Base class of all features.

    A feature produces one or more numerical values. These values
    are stored in so-called fields.

    Args:
        dimensions: int or [str] or None, either the number of dimensions this feature
            or a list of their names.

            If dimensions is an integer, the dimension names will be indexes from 0 to `dimensions-1`.
            If dimensions is None, the number of dimensions and their names will be
            determined dynamically.

    Attributes:
        dimensions: [int|str] or None, the names of the fields produced by this feature.
    """

    def __init__(self, dimensions=None):
        if hasattr(self, "Dimensions"):
            dimensions = self.Dimensions
        if type(dimensions) is int:
            dimensions = list(range(dimensions))
        self.dimensions = sorted(set(dimensions)) if dimensions else None
        self.slot = {}
        self.rows = []

    def push(self):
        if self.dimensions:
            for dimension, value in iterate_items(self.slot):
                if dimension not in self.dimensions:
                    raise KeyError("unknown dimension '{}' (have dimensions {})".format(dimension, self.dimensions))
        self.rows.append(self.slot)
        self.slot = {}

    def array(self):
        # If dimensions is None, determine the dimensions by looking at the data.
        if self.dimensions is None:
            self.dimensions = sorted({dimension for row in self.rows for dimension, value in iterate_items(row)})
        result = Array(columns=self.dimensions)
        for i, row in enumerate(self.rows):
            values = [0.0] * len(self.dimensions)
            for dimension, value in iterate_items(row):
                values[self.dimensions.index(dimension)] = value
            result.data.append(values)
        return result


class Numerical(Feature, CurriedSet):
    """Produces a single numerical value."""

    def set(self, *args):
        index = 0
        if len(args) == 1:
            value, = args
        else:
            index, value = args
        self.slot[index] = value


class Categorical(Feature, CurriedSet):
    """Performs one hot encoding on categorical data.

    Args:
        values: list, the values to encode.
    """

    def __init__(self, values):
        super().__init__(dimensions=values)

    def set(self, token, weight=1.0):
        if token in self.dimensions:
            self.slot[token] = weight


class Hashed(Feature):
    """Hashes arbitrary values into a fixed number of buckets.

    Args:
        hash: callable, the hash function to map features to buckets.
            Please note that Python's hash() function is randomized. Using it
            here maps values to different buckets on each program run.
        buckets: int, the number of buckets to use.
        replace: str or callable, defines a replacement strategy if a value
            gets assigned to a non-empty bucket. Supported strategies are
                - 'sum', which adds the value to bucket
                - 'max', which stores the maximum in the bucket
            Additionally, a callable function can be passed to implement
            a custom replacement strategy.
        random_sign: bool, if True, makes the sign of the weights depend on
            the hash values.
    """

    def __init__(self, hash=fnv32a, buckets=100, replace=None, random_sign=False):
        super().__init__(dimensions=buckets)
        self.buckets = buckets
        self.random_sign = random_sign
        self.replace = None
        if replace == "sum":
            self.func = lambda new, old: new + old
        if replace == "max":
            self.func = lambda new, old: max(new, old)
        if callable(replace):
            self.func = replace
        self.hash = hash

    def set(self, token, weight=1.0):
        key = self.hash(token)
        if self.random_sign:
            if key & 0x80000000 != 0:
                weight = -weight
        index = key % self.buckets
        if self.replace is not None and index in self.slot:
            weight = self.replace(weight, self.slot[index])
        self.slot[index] = weight
