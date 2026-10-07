import pytest
from calc.ops import add, divide, mean, subtract


def test_add():
    assert add(2, 3) == 5


def test_subtract():
    assert subtract(5, 3) == 2


class TestDivide:
    def test_divide_ok(self):
        assert divide(6, 3) == 2

    def test_divide_by_zero(self):
        with pytest.raises(ValueError, match="division by zero"):
            divide(1, 0)


@pytest.mark.parametrize(("xs", "expected"), [([1, 2, 3], 2), ([10, 20], 15)])
def test_mean(xs, expected):
    assert mean(xs) == expected


@pytest.mark.skip(reason="intentionally skipped")
def test_skipped():
    raise AssertionError("this test is skipped and must never run")
