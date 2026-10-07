"""Tests for resolve_test_names: mapping user-given test names to collected node ids."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fixtures import copy_sample_repo

from ci_fix.tools.pytest_runner import PytestRunner, ResolvedTests, resolve_test_names

COLLECTED = [
    "tests/test_a.py::test_one",
    "tests/test_a.py::test_shared",
    "tests/test_a.py::TestThing::test_method",
    "tests/test_a.py::test_m[1]",
    "tests/test_a.py::test_m[2]",
    "tests/test_b.py::test_shared",
    "tests/test_b.py::TestOther::test_method",
    "tests/sub/test_c.py::test_unique",
]


def _resolve(*names: str) -> ResolvedTests:
    return resolve_test_names(list(names), COLLECTED)


def test_full_id_kept() -> None:
    r = _resolve("tests/test_a.py::test_one")
    assert r.node_ids == ["tests/test_a.py::test_one"]
    assert not r.ambiguous
    assert not r.not_found


def test_full_id_of_parametrized_base_kept() -> None:
    r = _resolve("tests/test_a.py::test_m")
    assert r.node_ids == ["tests/test_a.py::test_m"]
    assert not r.not_found


def test_full_id_not_collected() -> None:
    r = _resolve("tests/test_a.py::test_missing")
    assert r.node_ids == []
    assert r.not_found == ["tests/test_a.py::test_missing"]


def test_bare_unique_name() -> None:
    r = _resolve("test_unique")
    assert r.node_ids == ["tests/sub/test_c.py::test_unique"]


def test_bare_ambiguous_name() -> None:
    r = _resolve("test_shared")
    assert r.node_ids == []
    assert r.ambiguous == {
        "test_shared": ["tests/test_a.py::test_shared", "tests/test_b.py::test_shared"]
    }
    assert not r.not_found


def test_class_method_partial() -> None:
    r = _resolve("TestThing::test_method")
    assert r.node_ids == ["tests/test_a.py::TestThing::test_method"]


def test_bare_method_in_two_classes_is_ambiguous() -> None:
    r = _resolve("test_method")
    assert r.node_ids == []
    assert r.ambiguous["test_method"] == [
        "tests/test_a.py::TestThing::test_method",
        "tests/test_b.py::TestOther::test_method",
    ]


def test_bare_parametrized_collapses_to_base() -> None:
    r = _resolve("test_m")
    assert r.node_ids == ["tests/test_a.py::test_m"]
    assert not r.ambiguous


def test_name_with_param_matches_exact_case() -> None:
    r = _resolve("test_m[2]")
    assert r.node_ids == ["tests/test_a.py::test_m[2]"]


def test_full_id_with_param_kept() -> None:
    r = _resolve("tests/test_a.py::test_m[1]")
    assert r.node_ids == ["tests/test_a.py::test_m[1]"]


def test_unknown_name() -> None:
    r = _resolve("test_nope")
    assert r.node_ids == []
    assert r.not_found == ["test_nope"]


def test_unknown_param() -> None:
    r = _resolve("test_m[99]")
    assert r.node_ids == []
    assert r.not_found == ["test_m[99]"]


def test_duplicates_deduped_order_preserved() -> None:
    r = _resolve(
        "test_unique",
        "tests/test_a.py::test_one",
        "tests/sub/test_c.py::test_unique",  # same id as the bare name above
        "test_one",
        "test_m",
    )
    assert r.node_ids == [
        "tests/sub/test_c.py::test_unique",
        "tests/test_a.py::test_one",
        "tests/test_a.py::test_m",
    ]


def test_mixed_buckets() -> None:
    r = _resolve("test_one", "test_shared", "test_ghost")
    assert r.node_ids == ["tests/test_a.py::test_one"]
    assert list(r.ambiguous) == ["test_shared"]
    assert r.not_found == ["test_ghost"]


def test_empty_inputs() -> None:
    r = resolve_test_names([], COLLECTED)
    assert r.node_ids == [] and not r.ambiguous and r.not_found == []
    r = resolve_test_names(["test_one"], [])
    assert r.not_found == ["test_one"]


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    return copy_sample_repo(tmp_path / "sample_repo")


def test_resolve_against_collected_sample_repo(sample_repo: Path, tmp_path: Path) -> None:
    runner = PytestRunner(sample_repo, Path(sys.executable), tmp_path / "reports")
    collected = runner.collect()
    r = resolve_test_names(["test_subtract", "test_add", "test_divide_by_zero"], collected)
    assert r.node_ids == [
        "tests/test_ops.py::test_subtract",
        "tests/test_ops.py::TestDivide::test_divide_by_zero",
    ]
    assert r.ambiguous == {
        "test_add": ["tests/test_errors.py::test_add", "tests/test_ops.py::test_add"]
    }
    assert r.not_found == []
