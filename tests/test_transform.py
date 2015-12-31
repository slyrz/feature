import random

from feature import *


def test_transform():
    """Test if custom features work."""

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
