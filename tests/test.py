from collections import Counter

from feature import *


def test_setter():
    """Test if setting features works."""

    builder = Builder({"a": Numerical(), })

    builder.set_a(1)
    builder.push()
    builder.set("a", 2)
    builder.push()
    builder.set(3)
    builder.push()

    array = builder.array()
    assert len(array) == 3
    assert all(tuple(row) == (i, ) for i, row in enumerate(array, 1))


def test_setter_extended():
    """Test if setting multiple features works."""

    builder = Builder({
        "a": Numerical(),
        "b": Numerical(),
        "c": Builder({
            "d": Numerical(),
        }),
    })

    builder.set_a(10)
    builder.set_b(20)
    builder.set_c_d(30)
    builder.push()

    # Since `d` is the only feature in `c`, the name can be omitted.
    builder.set_a(11)
    builder.set_b(21)
    builder.set_c(31)
    builder.push()

    builder.set("a", 12)
    builder.set("b", 22)
    builder.set("c", "d", 32)
    builder.push()

    builder.set("a", 13)
    builder.set("b", 23)
    builder.set("c", 33)
    builder.push()

    array = builder.array()
    assert array.shape == (4, 3)
    assert all(tuple(row) == (10 + i, 20 + i, 30 + i) for i, row in enumerate(array))


def test_numerical_feature():
    """Test the Numerical feature class."""

    builder = Builder({"a": Numerical(), "b": Numerical(), })

    for i in range(10):
        builder.set_a(i)
        builder.set_b(i * 10)
        builder.push()

    array = builder.array()
    assert array.shape == (10, 2)

    count = Counter()
    for row in array:
        for column, value in zip(array.columns, row):
            count[column[0]] += value
    assert count["a"] == 45 and count["b"] == 450


def test_categorical_feature():
    """Test the Categorical feature class."""

    builder = Builder({
        "a": Categorical([i for i in range(3)]),
        "b": Categorical([i for i in range(5)]),
    })

    for i in range(10):
        builder.set_a(i % 3)
        builder.set_b(i % 5)
        builder.push()

    array = builder.array()
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

    builder = Builder({
        "a": Hashed(size=3, hash=mock),
        "b": Hashed(size=5, hash=mock),
    })

    for i in range(10):
        builder.set_a("abcde"[i % 3])
        builder.set_b("abcde"[i % 5])
        builder.push()

    array = builder.array()
    assert array.shape == (10, 8)

    for i, row in enumerate(array):
        for column, value in zip(array.columns, row):
            feature, index = column.split("_")
            if feature == "a":
                assert value == float((i % 3) == int(index))
            else:
                assert value == float((i % 5) == int(index))
