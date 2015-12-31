import random
import string
from collections import Counter
from nose.tools import assert_raises
from importlib import import_module

from feature import *


def test_setter_extended():
    """Test if setting multiple features works."""

    group = Group({
        "a": Numerical(),
        "b": Numerical(),
        "c": Group({
            "d": Numerical(),
        }),
    })

    group.set_a(10)
    group.set_b(20)
    group.set_c_d(30)
    group.push()

    group.set_a(11)
    group.set_b(21)
    group.set_c(31)
    group.push()

    group.set("a", 12)
    group.set("b", 22)
    group.set("c", "d", 32)
    group.push()

    group.set("a", 13)
    group.set("b", 23)
    group.set("c", 33)
    group.push()

    array = group.array()
    assert array.shape == (4, 3)
    for i, row in enumerate(array):
        assert tuple(row) == (10 + i, 20 + i, 30 + i)


def test_numerical_feature():
    """Test the Numerical feature class."""

    group = Group({
        "a": Numerical(),
        "b": Numerical(),
        "c": Numerical(fields=3),
        "d": Numerical(fields="xyz"),
    })

    group.set_a(100)
    group.set_b(200)
    group.set_c(0, 10)
    group.set_c(1, 20)
    group.set_c(2, 30)
    group.set_d("x", 1)
    group.set_d("y", 2)
    group.set_d("z", 3)
    group.push()

    group.set_a(100)
    group.set_b(200)
    group.set_c_0(40)
    group.set_c_1(50)
    group.set_c_2(60)
    group.set_d_x(1)
    group.set_d_y(2)
    group.set_d_z(3)
    group.push()

    array = group.array()
    assert array.shape == (2, 8)

    count = Counter()
    for row in array:
        for column, value in zip(array.columns, row):
            count[column[0]] += value

    assert count["a"] == 200
    assert count["b"] == 400
    assert count["c"] == 210
    assert count["d"] == 12


def test_categorical_feature():
    """Test the Categorical feature class."""

    group = Group({
        "a": Categorical(list(range(3))),
        "b": Categorical(list(range(5))),
    })

    for i in range(10):
        group.set_a(i % 3)
        group.set_b(i % 5)
        group.push()

    array = group.array()
    assert array.shape == (10, 8)

    for i, row in enumerate(array):
        for column, value in zip(array.columns, row):
            feature, index = column.split("_")
            if feature == "a":
                assert value == float((i % 3) == int(index))
            else:
                assert value == float((i % 5) == int(index))


def test_hashed_feature():
    """Test the Hashed feature class."""

    def mock(c):
        return ord(c) - ord('a')

    group = Group({
        "a": Hashed(size=3, hash=mock),
        "b": Hashed(size=5, hash=mock),
    })

    for i in range(10):
        group.set_a("abcde" [i % 3])
        group.set_b("abcde" [i % 5])
        group.push()

    array = group.array()
    assert array.shape == (10, 8)

    for i, row in enumerate(array):
        for column, value in zip(array.columns, row):
            feature, index = column.split("_")
            if feature == "a":
                assert value == float((i % 3) == int(index))
            else:
                assert value == float((i % 5) == int(index))


def test_stress():
    """Test to see if using different classes works."""

    group = Group({
        "a": Numerical(),
        "b": Numerical(),
        "c": Categorical(list(range(5))),
        "d": Hashed(size=5),
        "e": Hashed(size=5, random_sign=True),
    })

    for i in range(100):
        group.set_a(random.random())
        group.set_b(random.random())
        group.set_c(random.randint(0, 4))
        for i in range(10):
            group.set_d("".join(random.sample(string.ascii_lowercase, 10)))
            group.set_e("".join(random.sample(string.ascii_lowercase, 10)))
        group.push()

    array = group.array()
    assert array.shape == (100, 17)


def test_transform():
    """Test if transforming the array works."""

    def transform(array):
        """Turns the (n,2) array into a (n,4) array."""
        new = Array(columns="abcd")
        for _ in array:
            new.append([1, 2, 3, 4])
        return new

    group = Group({"a": Numerical(), "b": Numerical()}, transform=transform)

    for _ in range(10):
        group.set_a(random.random())
        group.set_b(random.random())
        group.push()

    array = group.array()
    assert array.shape == (10, 4)


class CustomSized(Feature):
    """Custom feature with predefined size."""

    Fields = 4

    def set(self, x):
        self.slot[x] = 1.0


class CustomNamed(Feature):
    """Custom feature with predefined field names."""

    Fields = ["a", "b", "c", "d"]

    def set(self, x):
        self.slot[x] = 1.0


class CustomDynamic(Feature):
    """Custom feature with dynamic field names."""

    def set(self, x):
        self.slot[x] = 1.0


def test_custom_features():
    """Test if custom features work."""

    group = Group({
        "a": CustomSized(),
        "b": CustomNamed(),
        "c": CustomDynamic(),
    })

    for _ in range(10):
        for x in range(4):
            group.set_a(x)
        for x in "abcd":
            group.set_b(x)
            group.set_c(x)
        group.push()

    array = group.array()
    assert array.shape == (10, 12)


def test_field_name_errors():
    """Test if using undefined keys in features with predefined size or
    field names causes an exception."""

    group = Group({"a": CustomSized(), "b": CustomNamed(), })
    assert_raises(KeyError, group.set_a, 5)
    assert_raises(KeyError, group.set_b, "z")


def test_custom_empty():
    """Test if array can be build from empty features when the field size or
    the field names are fixed."""

    group = Group({
        "a": CustomSized(),
        "b": CustomNamed(),
        "c": Numerical(fields=4),
        "d": Hashed(size=4),
        "e": Categorical([1, 2, 3, 4]),
    })

    for i in range(10):
        group.push()

    array = group.array()
    assert array.shape == (10, 20)


def test_array_concatenate():
    """Test if array concatenation works."""

    array = Array(columns="abc")
    for i in range(10):
        array.append([1, 2, 3])

    # Any 2-dimensional array witht the same number of rows should work.
    other = [[4, 5, 6]] * len(array)
    array.concatenate(other)

    assert array.shape == (10, 6)
    assert len(array.columns) == 6
    assert all(type(column) is str for column in array.columns)

    # Now this should fail since the columns have the same names.
    other = Array(columns="abc")
    for i in range(10):
        other.append([1, 2, 3])
    assert_raises(ValueError, array.concatenate, other)

    # Adding a prefix should make it work.
    array.concatenate(other, prefix="other")
    assert array.shape == (10, 9)
    assert len(array.columns) == 9


class MockSklearnModel(object):
    def fit_transform(self, x):
        return [[1, 2]] * len(x)


class MockKerasModel(object):
    def predict(self, x):
        return [[1, 2]] * len(x)


def check_model_transform(transform, model):
    group = Group({
        "a": Numerical(fields=10),
        "b": Group({
                "c": Numerical(fields=10),
            }, transform=transform(model)),
    })

    for i in range(10):
        for j in range(10):
            group.set_a(j, random.random())
            group.set_b(j, random.random())
        group.push()

    array = group.array()
    assert array.shape == (10, 12)
    for row in array:
        assert tuple(row[-2:]) == (1, 2)


def test_sklearn_plugin():
    from feature.plugin.sklearn import Transform
    check_model_transform(Transform, MockSklearnModel())


def test_keras_plugin():
    from feature.plugin.keras import Transform
    check_model_transform(Transform, MockKerasModel())
