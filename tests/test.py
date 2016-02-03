from random import random, choice, randint
from string import printable
from collections import Counter
from nose.tools import assert_raises
from importlib import import_module

from feature import *


def randstr(length=12):
    return "".join([choice(printable) for _ in range(length)])


def test_setter_extended():
    """Test if setting multiple features works."""

    group = Group({"a": Numerical(), "b": Numerical(), "c": Group({"d": Numerical(), }), })

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

    group = Group({"a": Numerical(), "b": Numerical(), "c": Numerical(fields=3), "d": Numerical(fields="xyz"), })

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
    group.set_c(0, 40)
    group.set_c(1, 50)
    group.set_c(2, 60)
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

    group = Group({"a": Categorical("abc"), "b": Categorical("abcde"), })

    for i in range(10):
        group.set_a("abcde" [i % 3])
        group.set_b("abcde" [i % 5])
        # Ignore unkown elements.
        group.set_a("x")
        group.set_b("x")
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

    group = Group({"a": Hashed(size=3, hash=mock), "b": Hashed(size=5, hash=mock), })

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


def test_hashed_feature_random_sign():
    """Test if the default hash function distributes random signs evenly."""

    group = Group({"a": Hashed(size=100, random_sign=True), })

    for i in range(100):
        for j in range(100):
            group.set(randstr(), weight=123)
        group.push()

    array = group.array()
    assert array.shape == (100, 100)

    pos, neg = 0, 0
    for row in array:
        for value in row:
            assert value == 0 or abs(value) == 123
            pos += int(value > 0)
            neg += int(value < 0)
    assert pos and neg and abs(pos - neg) < (pos + neg) * 0.1


def test_stress():
    """Test to see if using different classes works."""

    group = Group({
        "a": Numerical(),
        "b": Numerical(),
        "c": Categorical(list(range(5))),
        "d": Hashed(size=5),
        "e": Hashed(size=5,
                    random_sign=True),
    })

    for i in range(100):
        group.set_a(random())
        group.set_b(random())
        group.set_c(randint(0, 4))
        for i in range(10):
            group.set_d(randstr())
            group.set_e(randstr())
        group.push()

    array = group.array()
    assert array.shape == (100, 17)


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


class CustomSlotList(Feature):
    """Custom feature with its own slot."""

    def set(self):
        self.slot = [1, 2, 3]


class CustomSlotDict(Feature):
    """Custom feature with its own slot."""

    def set(self):
        self.slot = {"foo": 1, "bar": 2, "baz": 3}


def test_custom_features():
    """Test if custom features work."""

    group = Group({
        "a": CustomSized(),
        "b": CustomNamed(),
        "c": CustomDynamic(),
        "d": CustomSlotList(),
        "e": CustomSlotDict(),
    })

    for _ in range(10):
        for x in range(4):
            group.set_a(x)
        for x in "abcd":
            group.set_b(x)
        group.set_c("blub")
        group.set_d()
        group.set_e()
        group.push()

    array = group.array()
    assert array.shape == (10, 15)


def test_field_name_errors():
    """Test if using undefined keys in features with predefined size or
    field names causes an exception."""

    group = Group({"test": CustomSized(), })
    group.set_test(5)
    assert_raises(KeyError, group.push)

    group = Group({"test": CustomNamed(), })
    group.set_test("e")
    assert_raises(KeyError, group.push)


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
    for row in array:
        assert tuple(row) == (1, 2, 3, 4, 5, 6)

    # Now this should fail since the columns have the same names.
    other = Array(columns="abc")
    for i in range(10):
        other.append([7, 8, 9])
    assert_raises(ValueError, array.concatenate, other)

    # Adding a prefix should make it work.
    array.concatenate(other, prefix="other")
    assert array.shape == (10, 9)
    assert len(array.columns) == 9
    for row in array:
        assert tuple(row) == (1, 2, 3, 4, 5, 6, 7, 8, 9)


def test_array_concatenate_numpy():
    try:
        import numpy
    except ImportError as e:
        return

    array = Array(columns="abc")
    for i in range(10):
        array.append([1, 2, 3])

    other = numpy.random.rand(len(array), 4)
    array.concatenate(other, prefix="other")
    assert array.shape == (10, 7)
    assert len(array.columns) == 7

    for i, row in enumerate(array):
        assert all(x == y for x, y in zip(row[-4:], other[i]))


def test_pipe_simple():
    """Test if transforming the array works."""

    def transform(array):
        """Turns the (n,2) array into a (n,4) array."""
        assert array.shape == (10, 2)
        new = Array(columns="abcd")
        for x, y in array:
            new.append([x, y, x + y, x * y])
        return new

    group = Pipe(Group({"a": Numerical(), "b": Numerical()}), transform)
    for _ in range(10):
        group.set_a(1e-6 + random())
        group.set_b(1e-6 + random())
        group.push()

    array = group.array()
    assert array.shape == (10, 4)

    for row in array:
        assert row[0] > 0.0 and row[1] > 0.0
        assert row[2] == row[0] + row[1]
        assert row[3] == row[0] * row[1]


def test_pipe_numpy():
    try:
        import numpy
        import numpy.testing
    except ImportError as e:
        return

    zero_mean = lambda x: x - x.mean(axis=0)
    unit_variance = lambda x: x / x.std(axis=0)

    group = Group({
        "a": Numerical(fields=10),
        "b": Pipe(Numerical(fields=10), numpy.array, zero_mean, unit_variance),
    })

    for i in range(200):
        for j in range(10):
            group.set_a(j, random())
            group.set_b(j, random())
        group.push()

    array = numpy.array(group.array())
    assert array.shape == (200, 20)

    a_avg, a_std = 0.5, 0.2887
    b_avg, b_std = 0.0, 1.0

    avg = array.mean(axis=0)
    numpy.testing.assert_allclose(avg, [a_avg] * 10 + [b_avg] * 10, atol=0.2, rtol=0.0)

    std = array.std(axis=0)
    numpy.testing.assert_allclose(std, [a_std] * 10 + [b_std] * 10, atol=0.1, rtol=0.0)
