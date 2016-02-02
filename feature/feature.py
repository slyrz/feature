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
    return [ str(i) for i in range(len(array[0])) ]


def iterate_items(obj):
    """Iterates over the object's key value pairs (key, value), where obj[key] == value."""
    if hasattr(obj, "items"):
        return obj.items()
    # If the object lacks the items() function, assume it's a list with keys [0, len(obj)-1].
    return enumerate(obj)


class Array(UserList):
    """Array stores the real-valued features.

    Args:
        length: int, create the given number of rows.

    Attributes:
        data: [[float]], 2-dimensional array with the numerical
            values of all features.
        columns: [str], the name of each column in data.
    """

    def __init__(self, length=0, columns=None):
        super().__init__()
        self._columns = list(columns) if columns is not None else []
        self.data = [[] for i in range(length)]
        # TODO: replace this class with pandas DataFrame?

    def concatenate(self, other, prefix=""):
        """Concatenates the columns from the `other` to `self`. `other` can be
        of any type that supports len, indexing and enumeration. Furthermore
        column names can be supplied in a `columns` attribute.

        Args:
            other: contains the new columns.
            prefix: str, optional prefix added to the new column names
                to avoid name clashes.
        """
        if not self.data:
            self.data = [[] for i in range(len(other))]

        if len(self) != len(other):
            raise ValueError("array length does not match - have {} and {}".format(len(self), len(other)))

        old_columns = self.columns
        new_columns = other.columns if hasattr(other, "columns") else numbered_columns(other)
        if prefix:
            new_columns = [ "{}_{}".format(prefix, name) for name in new_columns ]

        # Make sure the column names do not clash.
        for column in new_columns:
            if column in old_columns:
                raise ValueError("a column named '{}' already exists".format(column))

        self._columns = old_columns + new_columns
        for i, row in enumerate(other):
            self.data[i].extend(row)

    @property
    def shape(self):
        """Returns the array shape."""
        return (len(self.data), len(self.data[0]))

    @property
    def columns(self):
        if not self._columns:
            self._columns = numbered_columns(self)
        return self._columns


class Store(object):

    def push(self):
        raise NotImplementedError()

    def array(self):
        raise NotImplementedError()


class Group(Store):
    """Group produces real-valued feature arrays from one or more
    Feature/Group classes.

    Args:
        features: {str: Feature|Group}, instances of Feature/Group classes
            stored under their names.
        transform: callable, function called to transform arrays before
            returning them in the array() function.
    """

    def __init__(self, features, transform=None):
        if transform is not None:
            self.transform = transform
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

    def transform(self, array):
        return array

    def array(self):
        result = Array()
        for name, feature in sorted(self.features.items()):
            result.concatenate(feature.array(), prefix=name)
        result = self.transform(result)
        return result

    def __getattr__(self, name):
        function, *partial_applied_args = name.split("_")
        if function != "set" or not partial_applied_args:
            return self.__getattribute__(name)
        return partial(self.set, *partial_applied_args)


class Feature(Store):
    """Base class of all features.

    A feature produces one or more numerical values. These values
    are stored in so-called fields.

    Args:
        fields: int or [int|str] or None, either the number of fields this feature
            produces or a list of their names/indexes.
            If fields is an integer, the field names will be indexes
            from 0 to `fields-1`.
            If fields is None, the number of fields and their names will be
            determined dynamically.

    Attributes:
        fields: [int|str] or None, the names of the fields produced by this feature.
    """

    def __init__(self, fields=None):
        if hasattr(self, "Fields"):
            fields = self.Fields
        if type(fields) is int:
            fields = list(range(fields))
        self.fields = list(fields) if fields else None
        self.slot = {}
        self.rows = []

    def push(self):
        if self.fields:
            for field, _ in iterate_items(self.slot):
                if field not in self.fields:
                    raise KeyError("unknown field '{}'".format(field))
        self.rows.append(self.slot)
        self.slot = {}

    def array(self):
        # If fields wasn't set, we have to determine the fields based on the provided rows.
        if self.fields is None:
            self.fields = sorted({ field for row in self.rows for field, value in iterate_items(row) })

        result = Array(columns=self.fields)
        for i, row in enumerate(self.rows):
            values = [0.0] * len(self.fields)
            for field, value in iterate_items(row):
                values[self.fields.index(field)] = value
            result.data.append(values)
        return result

    def set(self, value):
        raise NotImplementedError()


class Numerical(Feature):
    """Produces a single numerical value."""

    def __init__(self, fields=1):
        super().__init__(fields=fields)

    def set(self, *args):
        index = 0
        if len(args) == 1:
            value, = args
        else:
            index, value = args
        self.slot[index] = value


class Categorical(Feature):
    """Performs one hot encoding on categorical data.

    Args:
        values: list, the values to encode.
    """

    def __init__(self, values):
        values = sorted(set(values))
        super().__init__(fields=len(values))
        self.values = values

    def set(self, token, weight=1.0):
        if token in self.values:
            self.slot[self.values.index(token)] = weight


class Hashed(Feature):
    """Hashes arbitrary values into a fixed number of buckets.

    Args:
        hash: callable, the hash function to map features to buckets.
            Please note that Python's hash() function is randomized. Using it
            here maps values to different buckets on each program run.
        size: int, the number of buckets to use.
        replace: str or callable, defines a replacement strategy if a value
            gets assigned to a non-empty bucket. Supported strategies are
                - 'sum', which adds the value to bucket
                - 'max', which stores the maximum in the bucket
            Additionally, a callable function can be passed to implement
            a custom replacement strategy.
        random_sign: bool, if True, makes the sign of the weights depend on
            the hash values.
    """

    def __init__(self, hash=fnv32a, size=100, replace=None, random_sign=False):
        super().__init__(fields=size)
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
        index = key % len(self.fields)
        if self.replace is not None and index in self.slot:
            weight = self.replace(weight, self.slot[index])
        self.slot[index] = weight
