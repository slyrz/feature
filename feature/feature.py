import collections


def fnv32a(text):
    h = 0x811c9dc5
    for c in text:
        h = ((h ^ ord(c)) * 0x01000193) & 0xffffffff
    return h


class Builder(object):
    """Builder produces real-valued feature arrays from one or more
    Feature/Builder classes.

    Args:
        features: {str: Feature|Builder}, instances of Feature/Builder classes
            stored under their names.
    """

    def __init__(self, features):
        self._features = features
        self._slots = {}
        self._rows = []

    def set(self, *args, **kwargs):
        # If builder contains only a single feature, allow omitting the name.
        # Otherwise name should be the first argument and the remaining arguments
        # get passed on to the feature's set() function.
        if len(args) == 1:
            name, = self._features.keys()
        else:
            name = args[0]
            args = args[1:]

        feature = self._features[name]
        # If the feature is an instance of Builder, let it take care of storing
        # its values.
        # Otherwise create a slot that stores the value in this class.
        if isinstance(feature, Builder):
            feature.set(*args, **kwargs)
        else:
            if name not in self._slots:
                feature.slot = self._slots[name] = Slot(feature.fields)
            feature.set(*args, **kwargs)

    def push(self):
        # To keep the number of rows across all nested builders in sync,
        # we have to inform them that a new row is being added.
        for feature in self._features.values():
            if isinstance(feature, Builder):
                feature.push()
        self._rows.append(self._slots)
        self._slots = {}

    def _get_fields(self, name):
        feature = self._features[name]
        if feature.fields is not None:
            fields = set(feature.fields)
        else:
            fields = set()
            for row in self._rows:
                if name not in row: continue
                fields |= set(row[name].keys())
        return sorted(fields)

    def _array_from_feature(self, name, feature):
        result = Array()
        result.columns = self._get_fields(name)
        for i, row in enumerate(self._rows):
            values = [0.0] * len(result.columns)
            if name in row:
                for field, value in row[name].items():
                    values[result.columns.index(field)] = value
            result.data.append(values)
        return result

    def _array_from_builder(self, name, feature):
        return feature.array()

    def array(self):
        result = Array(length=len(self._rows))
        for name, feature in sorted(self._features.items()):
            if isinstance(feature, Feature):
                part = self._array_from_feature(name, feature)
            if isinstance(feature, Builder):
                part = self._array_from_builder(name, feature)
            result.concatenate(part, prefix=name)
        return result

    def _curry(self, func, parts):
        """Returns the curried function `func` with partially applied
        arguments `parts`.

        Mixing Haskell and Python syntax, the returned value is
        >>> (self.function *parts)
        """

        def closure(*args, **kwargs):
            args = list(parts) + list(args)
            return getattr(self, func)(*args, **kwargs)

        return closure

    def __getattr__(self, name):
        function, *partial_applied_args = name.split("_")
        if function != "set":
            raise AttributeError(function)
        return self._curry(function, partial_applied_args)


class Array(collections.UserList):
    """Array stores the real-valued features.

    Args:
        length: int, create the given number of rows.

    Attributes:
        data: [[float]], 2-dimensional array with the numerical
            values of all features.
        columns: [str], the name of each column in data.
    """

    def __init__(self, length=0):
        super().__init__()
        self.columns = []
        self.data = [[] for i in range(length)]
        # TODO: replace this class with pandas DataFrame?

    def concatenate(self, other, prefix=""):
        """Concatenates the columns from the `other` Array to `self`.

        Args:
            other: Array, contains the new columns.
            prefix: str, optional prefix added to the new column names
                to avoid name clashes.
        """
        assert len(self) == len(other)

        if prefix:
            self.columns.extend("{}_{}".format(prefix, name) for name in other.columns)
        else:
            self.columns.extend(other.columns)

        for i, row in enumerate(other.data):
            self.data[i].extend(row)


class Slot(collections.UserDict):
    """Slot stores the numerical values produced by a Feature class.

    For each row, a Feature class writes the values of its fields into a
    Slot. The Builder stores these Slots and uses them to produce Arrays.

    Args:
        fields: [int|str], if given, all keys passed to the Slot's item setter
            must be members of this list.
    """

    def __init__(self, fields=None):
        super().__init__()
        self.fields = fields

    def __setitem__(self, key, value):
        if self.fields is not None:
            assert key in self.fields
        self.data[key] = float(value)


class Feature(object):
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
        self.slot = None
        self.fields = fields

    def set(self, value):
        self.slot[0] = value


class Numerical(Feature):
    pass


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
        self.slot[self.values.index(token)] = weight


class Hashed(Feature):
    """Hashes arbitrary values into a fixed number of buckets.

    Args:
        hash: callable, the hash function to map features to buckets.
            Please note that Python's hash() function is randomized. Using it
            here maps values to different buckets on each program run.
        size: int, the number of buckets to use.
        additive: bool, if True, adds up weights mapped to the same bucket.
        random_sign: bool, if True, makes the sign of the weights depend on
            the hash values.
    """

    def __init__(self, hash=fnv32a, size=100, additive=True, random_sign=False):
        super().__init__(fields=size)
        self.random_sign = random_sign
        self.additive = additive
        self.hash = hash

    def set(self, token, weight=1.0):
        key = self.hash(token)
        if self.random_sign:
            if key & 0x80000000 != 0:
                weight = -weight
        index = key % len(self.fields)
        if self.additive and index in self.slot:
            weight += self.slot[index]
        self.slot[index] = weight
