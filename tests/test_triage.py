"""Tests for ci_fix.triage (slice 7): traceback file extraction and failure grouping."""

from __future__ import annotations

from pathlib import Path

import pytest

from ci_fix.triage import group_failures, traceback_files

FILES = [
    "src/calc/ops.py",
    "src/calc/strings.py",
    "src/calc/util.py",
    "tests/test_ops.py",
    "tests/test_strings.py",
    "tests/test_util.py",
    "tests/test_misc.py",
    ".venv/lib/python3.11/site-packages/dep/core.py",
]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel in FILES:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# placeholder\n", encoding="utf-8")
    return root


# ---- traceback_files ------------------------------------------------------------------------

PYTEST_LONG = """\
def test_mean():
>       assert mean([1, 2, 3]) == 2

tests/test_ops.py:27: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

xs = [1, 2, 3]

    def mean(xs):
>       return sum(xs) / (len(xs) + 1)
E       assert 1.5 == 2

src/calc/ops.py:21: AssertionError
"""

PYTEST_SHORT = """\
tests/test_ops.py:27: in test_mean
    assert mean([1, 2, 3]) == 2
src/calc/ops.py:21: in mean
    return helper(xs)
src/calc/util.py:3: in helper
    raise ZeroDivisionError
E   ZeroDivisionError
"""


def test_pytest_long_format(repo: Path) -> None:
    assert traceback_files(PYTEST_LONG, repo) == {"tests/test_ops.py", "src/calc/ops.py"}


def test_pytest_short_format(repo: Path) -> None:
    assert traceback_files(PYTEST_SHORT, repo) == {
        "tests/test_ops.py",
        "src/calc/ops.py",
        "src/calc/util.py",
    }


def test_unittest_style_absolute_paths(repo: Path) -> None:
    details = f"""\
Traceback (most recent call last):
  File "{repo / "tests/test_strings.py"}", line 8, in test_shout
    self.assertEqual(shout("hi"), "HI!")
  File "{repo / "src/calc/strings.py"}", line 5, in shout
    return s.lower() + "!"
AssertionError: 'hi!' != 'HI!'
"""
    assert traceback_files(details, repo) == {"tests/test_strings.py", "src/calc/strings.py"}


def test_pytest_absolute_path_inside_repo_is_made_relative(repo: Path) -> None:
    details = f"{repo / 'src/calc/ops.py'}:11: in subtract\n    return a + b\nE   boom\n"
    assert traceback_files(details, repo) == {"src/calc/ops.py"}


def test_paths_outside_repo_are_ignored(repo: Path, tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere" / "lib.py"
    outside.parent.mkdir()
    outside.write_text("x = 1\n", encoding="utf-8")
    details = (
        f'  File "{outside}", line 1, in <module>\n'
        f"{outside}:1: in <module>\n"
        f'  File "/usr/lib/python3.11/json/decoder.py", line 355, in raw_decode\n'
        "src/calc/ops.py:11: AssertionError\n"
    )
    assert traceback_files(details, repo) == {"src/calc/ops.py"}


def test_site_packages_and_venv_are_ignored(repo: Path) -> None:
    venv_file = repo / ".venv/lib/python3.11/site-packages/dep/core.py"
    details = (
        f'  File "{venv_file}", line 1, in run\n'
        ".venv/lib/python3.11/site-packages/dep/core.py:1: in run\n"
        '  File "/opt/venv/lib/python3.11/site-packages/_pytest/python.py", line 1, in f\n'
        "/opt/venv/lib/python3.11/site-packages/pluggy/_hooks.py:513: in __call__\n"
        "tests/test_ops.py:9: AssertionError\n"
    )
    assert traceback_files(details, repo) == {"tests/test_ops.py"}


@pytest.mark.parametrize("details", ["", "assert 8 == 2", "failed: tests/test_ops.py::test_x"])
def test_no_traceback_lines_gives_empty_set(repo: Path, details: str) -> None:
    assert traceback_files(details, repo) == set()


def test_returns_a_set_without_duplicates(repo: Path) -> None:
    details = "src/calc/ops.py:11: in f\nsrc/calc/ops.py:12: in g\nsrc/calc/ops.py:13: E\n"
    result = traceback_files(details, repo)
    assert isinstance(result, set)
    assert result == {"src/calc/ops.py"}


# ---- group_failures -------------------------------------------------------------------------

A = "tests/test_ops.py::test_a"
A2 = "tests/test_ops.py::test_a2"
B = "tests/test_strings.py::test_b"
C = "tests/test_util.py::test_c"
D = "tests/test_misc.py::test_d"


def _tb(*files: str) -> str:
    return "".join(f"{f}:10: in something\n" for f in files)


def test_independent_failures_are_separate_groups(repo: Path) -> None:
    details = {
        A: _tb("tests/test_ops.py", "src/calc/ops.py"),
        B: _tb("tests/test_strings.py", "src/calc/strings.py"),
    }
    assert group_failures([A, B], details, repo) == [[A], [B]]


def test_shared_source_file_groups_tests(repo: Path) -> None:
    details = {
        A: _tb("tests/test_ops.py", "src/calc/ops.py"),
        B: _tb("tests/test_strings.py", "src/calc/ops.py"),
    }
    assert group_failures([A, B], details, repo) == [[A, B]]


def test_transitive_overlap_chains_into_one_group(repo: Path) -> None:
    details = {
        A: _tb("tests/test_ops.py", "src/calc/ops.py"),
        B: _tb("tests/test_strings.py", "src/calc/ops.py", "src/calc/strings.py"),
        C: _tb("tests/test_util.py", "src/calc/strings.py"),
        D: _tb("tests/test_misc.py", "src/calc/util.py"),
    }
    assert group_failures([A, B, C, D], details, repo) == [[A, B, C], [D]]


def test_same_test_file_is_one_group_even_without_shared_traceback_files(repo: Path) -> None:
    details = {A: _tb("src/calc/ops.py"), A2: _tb("src/calc/strings.py"), B: ""}
    assert group_failures([A, B, A2], details, repo) == [[A, A2], [B]]


def test_tests_without_traceback_files_are_singletons(repo: Path) -> None:
    details = {B: "", C: "assert False", D: ""}
    assert group_failures([B, C, D], details, repo) == [[B], [C], [D]]


def test_missing_details_entry_is_a_singleton(repo: Path) -> None:
    assert group_failures([B, C], {}, repo) == [[B], [C]]


def test_group_order_follows_first_member_and_members_keep_pending_order(repo: Path) -> None:
    # pending: B, A, D, C — B and C share util.py; A and D share ops.py.
    details = {
        B: _tb("tests/test_strings.py", "src/calc/util.py"),
        A: _tb("tests/test_ops.py", "src/calc/ops.py"),
        D: _tb("tests/test_misc.py", "src/calc/ops.py"),
        C: _tb("tests/test_util.py", "src/calc/util.py"),
    }
    assert group_failures([B, A, D, C], details, repo) == [[B, C], [A, D]]


def test_outside_and_venv_files_do_not_link_groups(repo: Path) -> None:
    shared = "/opt/venv/lib/python3.11/site-packages/_pytest/python.py:1: in call\n"
    details = {
        A: shared + _tb("tests/test_ops.py", "src/calc/ops.py"),
        B: shared + _tb("tests/test_strings.py", "src/calc/strings.py"),
    }
    assert group_failures([A, B], details, repo) == [[A], [B]]


def test_empty_pending(repo: Path) -> None:
    assert group_failures([], {}, repo) == []


def test_conftest_frames_do_not_link_groups(repo: Path) -> None:
    details = {
        A: _tb("tests/conftest.py", "tests/test_ops.py", "src/calc/ops.py"),
        B: _tb("tests/conftest.py", "tests/test_strings.py", "src/calc/strings.py"),
        C: _tb("conftest.py", "tests/test_util.py"),
    }
    assert group_failures([A, B, C], details, repo) == [[A], [B], [C]]


def test_traceback_files_still_reports_conftest(repo: Path) -> None:
    assert "tests/conftest.py" in traceback_files(_tb("tests/conftest.py"), repo)


def test_conftest_ignored_but_other_shared_file_still_links(repo: Path) -> None:
    details = {
        A: _tb("tests/conftest.py", "tests/test_ops.py", "src/calc/util.py"),
        B: _tb("tests/conftest.py", "tests/test_strings.py", "src/calc/util.py"),
    }
    assert group_failures([A, B], details, repo) == [[A, B]]
