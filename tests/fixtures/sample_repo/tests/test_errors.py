import pytest
from calc.ops import add


@pytest.fixture
def broken_resource():
    raise RuntimeError("fixture setup failed")


def test_fixture_error(broken_resource):
    assert broken_resource is not None


def test_add():
    """Same bare name as test_ops.py::test_add, so "test_add" alone is ambiguous."""
    assert add(-1, 1) == 0
