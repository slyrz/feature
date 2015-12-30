from nose.tools import assert_raises
from feature import *


class CustomSized(Feature):

    Fields = 4

    def set(self, x):
        self.slot[x] = 1.0


class CustomNamed(Feature):

    Fields = ["a", "b", "c", "d"]

    def set(self, x):
        self.slot[x] = 1.0


class CustomDynamic(Feature):
    def set(self, x):
        self.slot[x] = 1.0


def test_custom_features():
    """Test if custom features work."""

    builder = Builder({
        "a": CustomSized(),
        "b": CustomNamed(),
        "c": CustomDynamic(),
    })

    for _ in range(10):
        for x in range(4):
            builder.set_a(x)
        for x in "abcd":
            builder.set_b(x)
            builder.set_c(x)
        builder.push()

    array = builder.array()
    assert array.shape == (10, 12)


def test_field_name_errors():
    builder = Builder({"a": CustomSized(), "b": CustomNamed(), })

    assert_raises(KeyError, builder.set_a, 5)
    assert_raises(KeyError, builder.set_b, "z")
