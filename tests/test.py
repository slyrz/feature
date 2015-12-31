import random
import string
from collections import Counter

from feature import *


def test_setter():
    """Test if setting features works."""

    group = Group({"a": Numerical(), })

    group.set_a(1)
    group.push()
    group.set("a", 2)
    group.push()
    group.set(3)
    group.push()

    array = group.array()
    assert len(array) == 3
    assert all(tuple(row) == (i, ) for i, row in enumerate(array, 1))


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

    # Since `d` is the only feature in `c`, the name can be omitted.
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
    assert all(tuple(row) == (10 + i, 20 + i, 30 + i) for i, row in enumerate(array))


def test_numerical_feature():
    """Test the Numerical feature class."""

    group = Group({"a": Numerical(), "b": Numerical(), })

    for i in range(10):
        group.set_a(i)
        group.set_b(i * 10)
        group.push()

    array = group.array()
    assert array.shape == (10, 2)

    count = Counter()
    for row in array:
        for column, value in zip(array.columns, row):
            count[column[0]] += value
    assert count["a"] == 45 and count["b"] == 450


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
    """Test the Hashed feature class."""

    group = Group({
        "a": Numerical(),
        "b": Numerical(),
        "c": Categorical(list(range(5))),
        "d": Hashed(size=5, random_sign=True),
    })

    for i in range(100):
        group.set_a(random.random())
        group.set_b(random.random())
        group.set_c(random.randint(0, 4))
        for i in range(10):
            group.set_d("".join(random.sample(string.ascii_lowercase, 10)))
        group.push()

    array = group.array()
    assert array.shape == (100, 12)
