"""Tests for ci_fix.guards.patch_checker.check_patch (slice 5), written from the spec.

Each test builds a small git repo, commits the original files, rewrites some files in the
working tree and checks the report: one positive test per rule plus negatives that guard
against false positives on legitimate fixes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import git

from ci_fix.guards.patch_checker import ExpectationChange, PatchReport, Violation, check_patch
from ci_fix.tools.git import GitRepo

TEST_MATH = "tests/test_math.py"
TEST_UNIT = "tests/test_unit.py"
TEST_ERRORS = "tests/test_errors.py"
CONFTEST = "tests/conftest.py"
OPS = "src/calc/ops.py"
COMPAT = "src/calc/compat.py"

FILES: dict[str, str] = {
    TEST_MATH: """import pytest

from calc import add, divide, mean


def test_add():
    assert add(2, 3) == 5


def test_add_twice():
    total = add(1, 1)
    assert total == 2
    assert add(total, 1) == 3


def test_divide():
    assert divide(6, 3) == 2


def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)


def test_sum_values():
    data = [1, 2]
    total = sum(data)
    assert total == 3


def test_numbers(numbers):
    assert sum(numbers) == 6


@pytest.mark.parametrize(("xs", "expected"), [([1, 2, 3], 2), ([10, 20], 15)])
def test_mean(xs, expected):
    assert mean(xs) == expected


def test_close():
    assert mean([0.1, 0.2]) == pytest.approx(0.15, abs=1e-9)


@pytest.mark.skip(reason="already skipped before")
def test_already_skipped():
    assert add(1, 1) == 3


class TestGroup:
    def test_in_class(self):
        assert add(0, 0) == 0

    class TestNested:
        def test_nested(self):
            assert add(1, 2) == 3


def helper():
    return 42
""",
    TEST_UNIT: """import unittest

from calc import add


class ArithmeticChecks(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 6)

    def test_add_neg(self):
        self.assertEqual(add(-1, 1), 0)

    def test_close(self):
        self.assertAlmostEqual(add(0.1, 0.2), 0.3, places=7)
""",
    TEST_ERRORS: """import pytest
from calc.ops import add


@pytest.fixture
def broken_resource():
    raise RuntimeError("fixture setup failed")


def test_fixture_error(broken_resource):
    assert broken_resource is not None


def test_add():
    assert add(-1, 1) == 0
""",
    "tests/math_test.py": """from calc import add


def test_add_zero():
    assert add(0, 5) == 5
""",
    "tests/helpers.py": """def make_numbers():
    return [1, 2, 3]
""",
    CONFTEST: """import pytest


@pytest.fixture
def numbers():
    return [1, 2, 3]
""",
    "src/calc/__init__.py": '''"""Tiny calculator."""

from calc.ops import add, divide, mean

__all__ = ["add", "divide", "mean"]
''',
    OPS: """def add(a, b):
    return a + b


def divide(a, b):
    return a / b


def mean(xs):
    return sum(xs) / len(xs)


def load(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        raise
""",
    COMPAT: """import pytest  # legacy: used for approx in a helper


def approx(x):
    return pytest.approx(x)


def safe_int(text):
    try:
        return int(text)
    except Exception:
        return None
""",
    "pytest.ini": "[pytest]\naddopts = -q\n",
    "tox.ini": "[tox]\nenvlist = py311\n\n[testenv]\ncommands = pytest\n\n[pytest]\naddopts = -q\n",
    "setup.cfg": "[metadata]\nname = calc\n\n[tool:pytest]\naddopts = -q\n",
    "pyproject.toml": (
        '[project]\nname = "calc"\nversion = "0.1.0"\ndependencies = []\n\n'
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
    ),
    "README.md": "# calc\n",
}

Edit = tuple[str, str, str]  # (relpath, old, new); old must occur in the file


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    git("init", "-q", "-b", "main", cwd=root)
    git("config", "user.email", "tester@example.com", cwd=root)
    git("config", "user.name", "Tester", cwd=root)
    git("config", "commit.gpgsign", "false", cwd=root)
    for rel, text in FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    git("add", "-A", cwd=root)
    git("commit", "-q", "-m", "initial", cwd=root)
    return root


def apply(repo: Path, edits: list[Edit]) -> None:
    for rel, old, new in edits:
        path = repo / rel
        text = path.read_text(encoding="utf-8")
        assert old in text, f"{old!r} not in {rel}"
        path.write_text(text.replace(old, new, 1), encoding="utf-8")


def check(repo: Path, edits: list[Edit], explanation: str = "") -> PatchReport:
    apply(repo, edits)
    return check_patch(GitRepo(repo), explanation)


def rules(report: PatchReport) -> set[str]:
    return {v.rule for v in report.violations}


def assert_ok(report: PatchReport) -> None:
    assert report.ok, report.summary()
    assert report.violations == []


def assert_flags(report: PatchReport, rule: str) -> None:
    assert not report.ok
    assert rule in rules(report), report.summary()
    assert rule in report.summary()


# ---- no changes / legitimate changes ----------------------------------------------------------


def test_no_changes_is_ok(repo: Path) -> None:
    report = check_patch(GitRepo(repo), "")
    assert_ok(report)
    assert report.expectation_changes == []
    assert report.test_files_changed == []
    assert report.source_files_changed == []


def test_legit_source_fix_is_ok(repo: Path) -> None:
    report = check(repo, [(OPS, "return a + b", "return a - b")], "add was wrong")
    assert_ok(report)
    assert report.expectation_changes == []
    assert report.source_files_changed == [OPS]
    assert report.test_files_changed == []


def test_source_fix_raising_specific_error_is_ok(repo: Path) -> None:
    new = 'if b == 0:\n        raise ValueError("division by zero")\n    return a / b'
    assert_ok(check(repo, [(OPS, "return a / b", new)]))


def test_adding_a_new_test_is_ok(repo: Path) -> None:
    new_test = "\n\ndef test_add_negative():\n    assert add(-2, -3) == -5\n"
    report = check(repo, [(TEST_MATH, "\n\ndef helper():", new_test + "\n\ndef helper():")])
    assert_ok(report)
    assert report.expectation_changes == []
    assert report.test_files_changed == [TEST_MATH]


def test_adding_a_new_test_that_expects_an_exception_is_ok(repo: Path) -> None:
    new_test = (
        "\n\ndef test_divide_by_zero_again():\n"
        "    with pytest.raises(ZeroDivisionError):\n        divide(2, 0)\n"
    )
    assert_ok(check(repo, [(TEST_MATH, "\n\ndef helper():", new_test + "\n\ndef helper():")]))


def test_adding_a_new_test_file_is_ok(repo: Path) -> None:
    path = repo / "tests" / "test_new.py"
    path.write_text("from calc import add\n\n\ndef test_new():\n    assert add(1, 1) == 2\n")
    assert_ok(check_patch(GitRepo(repo), ""))


def test_adding_a_comment_in_a_test_is_ok(repo: Path) -> None:
    report = check(
        repo, [(TEST_MATH, "def test_add():\n", "def test_add():\n    # 2 + 3 is five\n")]
    )
    assert_ok(report)
    assert report.expectation_changes == []


def test_renaming_a_local_variable_in_a_test_is_ok(repo: Path) -> None:
    old = "    data = [1, 2]\n    total = sum(data)\n"
    new = "    values = [1, 2]\n    total = sum(values)\n"
    report = check(repo, [(TEST_MATH, old, new)])
    assert_ok(report)
    assert report.expectation_changes == []


def test_deleting_a_non_test_helper_is_ok(repo: Path) -> None:
    assert_ok(check(repo, [(TEST_MATH, "\n\ndef helper():\n    return 42\n", "\n")]))


def test_existing_skip_is_not_flagged(repo: Path) -> None:
    # The file already has a skipped test; an unrelated legit change must not trip skip_added.
    report = check(repo, [(TEST_MATH, "def test_add():\n", "def test_add():\n    # note\n")])
    assert_ok(report)


def test_adding_a_parametrize_case_is_ok(repo: Path) -> None:
    old = "[([1, 2, 3], 2), ([10, 20], 15)]"
    new = "[([1, 2, 3], 2), ([10, 20], 15), ([4], 4)]"
    assert_ok(check(repo, [(TEST_MATH, old, new)]))


def test_adding_a_fixture_to_conftest_is_ok(repo: Path) -> None:
    extra = "\n\n@pytest.fixture\ndef empty():\n    return []\n"
    path = repo / CONFTEST
    path.write_text(path.read_text() + extra)
    assert_ok(check_patch(GitRepo(repo), ""))


def test_non_python_change_is_ok(repo: Path) -> None:
    assert_ok(check(repo, [("README.md", "# calc\n", "# calc\n\nA calculator.\n")]))


def test_files_are_classified_as_test_or_source(repo: Path) -> None:
    edits = [
        (TEST_MATH, "def test_add():\n", "def test_add():\n    # c\n"),
        ("tests/math_test.py", "def test_add_zero():\n", "def test_add_zero():\n    # c\n"),
        (CONFTEST, "def numbers():\n", "def numbers():\n    # c\n"),
        (OPS, "def add(a, b):\n", "def add(a, b):\n    # c\n"),
        ("tests/helpers.py", "def make_numbers():\n", "def make_numbers():\n    # c\n"),
    ]
    report = check(repo, edits)
    assert_ok(report)
    # Every .py file in a test directory is a test file (helpers included).
    expected_tests = [TEST_MATH, "tests/math_test.py", CONFTEST, "tests/helpers.py"]
    assert sorted(report.test_files_changed) == sorted(expected_tests)
    assert report.source_files_changed == [OPS]


# ---- test_removed ----------------------------------------------------------------------------

ADD_TEST = "def test_add():\n    assert add(2, 3) == 5\n\n\n"


@pytest.mark.parametrize(
    "edits",
    [
        pytest.param([(TEST_MATH, ADD_TEST, "")], id="function-deleted"),
        pytest.param([(TEST_MATH, "def test_add():", "def test_add_renamed():")], id="renamed"),
        pytest.param([(TEST_MATH, "def test_add():", "def check_add():")], id="renamed-non-test"),
        pytest.param(
            [(TEST_MATH, "    def test_in_class(self):", "    def check_in_class(self):")],
            id="class-method",
        ),
        pytest.param(
            [(TEST_MATH, "        def test_nested(self):", "        def nested(self):")],
            id="nested-class-method",
        ),
        pytest.param(
            [(TEST_MATH, "    class TestNested:", "    class Nested:")], id="nested-class-renamed"
        ),
        pytest.param(
            [
                (
                    TEST_UNIT,
                    "    def test_add_neg(self):\n        self.assertEqual(add(-1, 1), 0)\n\n",
                    "",
                )
            ],
            id="unittest-method",
        ),
        pytest.param(
            [(TEST_UNIT, "    def test_add_neg(self):", "    def add_neg(self):")],
            id="unittest-renamed",
        ),
    ],
)
def test_removing_or_renaming_a_test_is_flagged(repo: Path, edits: list[Edit]) -> None:
    assert_flags(check(repo, edits), "test_removed")


def test_deleting_a_test_class_is_flagged(repo: Path) -> None:
    path = repo / TEST_MATH
    text = path.read_text()
    path.write_text(text[: text.index("class TestGroup:")] + "def helper():\n    return 42\n")
    assert_flags(check_patch(GitRepo(repo), ""), "test_removed")


def test_deleting_a_whole_test_file_is_flagged(repo: Path) -> None:
    (repo / TEST_UNIT).unlink()
    report = check_patch(GitRepo(repo), "")
    assert_flags(report, "test_removed")
    assert any(v.path == TEST_UNIT for v in report.violations)


# ---- skip_added ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "edits",
    [
        pytest.param(
            [(TEST_MATH, "def test_add():", "@pytest.mark.skip\ndef test_add():")], id="skip"
        ),
        pytest.param(
            [(TEST_MATH, "def test_add():", '@pytest.mark.skip(reason="x")\ndef test_add():')],
            id="skip-reason",
        ),
        pytest.param(
            [
                (
                    TEST_MATH,
                    "def test_add():",
                    '@pytest.mark.skipif(True, reason="x")\ndef test_add():',
                )
            ],
            id="skipif",
        ),
        pytest.param(
            [(TEST_MATH, "def test_add():", "@pytest.mark.xfail\ndef test_add():")], id="xfail"
        ),
        pytest.param(
            [
                (TEST_MATH, "import pytest\n", "import unittest\n\nimport pytest\n"),
                (TEST_MATH, "def test_add():", '@unittest.skip("x")\ndef test_add():'),
            ],
            id="unittest-skip-on-function",
        ),
        pytest.param(
            [(TEST_MATH, "class TestGroup:", "@pytest.mark.skip\nclass TestGroup:")],
            id="skip-on-class",
        ),
        pytest.param(
            [
                (
                    TEST_UNIT,
                    "    def test_add(self):",
                    '    @unittest.skip("x")\n    def test_add(self):',
                )
            ],
            id="unittest-skip",
        ),
        pytest.param(
            [
                (
                    TEST_UNIT,
                    "    def test_add(self):",
                    '    @unittest.skipIf(True, "x")\n    def test_add(self):',
                )
            ],
            id="unittest-skipIf",
        ),
        pytest.param(
            [
                (
                    TEST_UNIT,
                    "    def test_add(self):",
                    '    @unittest.skipUnless(False, "x")\n    def test_add(self):',
                )
            ],
            id="unittest-skipUnless",
        ),
        pytest.param(
            [
                (
                    TEST_UNIT,
                    "    def test_add(self):",
                    "    @unittest.expectedFailure\n    def test_add(self):",
                )
            ],
            id="unittest-expectedFailure",
        ),
        pytest.param(
            [(TEST_MATH, "def test_add():\n", 'def test_add():\n    pytest.skip("later")\n')],
            id="pytest.skip-call",
        ),
        pytest.param(
            [(TEST_MATH, "def test_add():\n", 'def test_add():\n    pytest.xfail("later")\n')],
            id="pytest.xfail-call",
        ),
        pytest.param(
            [
                (
                    TEST_MATH,
                    "def test_add():\n",
                    'def test_add():\n    pytest.importorskip("not_installed_mod")\n',
                )
            ],
            id="pytest.importorskip-call",
        ),
        pytest.param(
            [
                (
                    TEST_UNIT,
                    "    def test_add(self):\n",
                    '    def test_add(self):\n        self.skipTest("later")\n',
                )
            ],
            id="self.skipTest-call",
        ),
        pytest.param(
            [
                (
                    TEST_MATH,
                    "import pytest\n",
                    'import pytest\n\npytestmark = pytest.mark.skip(reason="x")\n',
                )
            ],
            id="pytestmark-skip",
        ),
        pytest.param(
            [(TEST_MATH, "import pytest\n", "import pytest\n\npytestmark = [pytest.mark.xfail]\n")],
            id="pytestmark-xfail-list",
        ),
    ],
)
def test_adding_a_skip_is_flagged(repo: Path, edits: list[Edit]) -> None:
    report = check(repo, edits)
    assert_flags(report, "skip_added")
    skip = next(v for v in report.violations if v.rule == "skip_added")
    assert isinstance(skip, Violation)
    assert skip.path in {TEST_MATH, TEST_UNIT}
    assert skip.message


# ---- assertion_removed / trivial_assertion / test_emptied ------------------------------------


@pytest.mark.parametrize(
    "edits",
    [
        pytest.param([(TEST_MATH, "    assert add(total, 1) == 3\n", "")], id="assert-stmt"),
        pytest.param(
            [
                (
                    TEST_MATH,
                    "    with pytest.raises(ZeroDivisionError):\n        divide(1, 0)\n",
                    "    divide(1, 1)\n",
                )
            ],
            id="pytest.raises",
        ),
        pytest.param(
            [(TEST_UNIT, "        self.assertEqual(add(-1, 1), 0)\n", "        add(-1, 1)\n")],
            id="assertEqual-call",
        ),
    ],
)
def test_removing_an_assertion_is_flagged(repo: Path, edits: list[Edit]) -> None:
    assert_flags(check(repo, edits), "assertion_removed")


def test_weakening_an_assertion_is_not_ok(repo: Path) -> None:
    # `assert total == 2` -> `assert total`: same count, changed source -> needs justification.
    report = check(repo, [(TEST_MATH, "    assert total == 2\n", "    assert total\n")])
    assert_flags(report, "unjustified_test_change")


@pytest.mark.parametrize(
    "edits",
    [
        pytest.param([(TEST_MATH, "assert add(2, 3) == 5", "assert True")], id="assert-True"),
        pytest.param([(TEST_MATH, "assert add(2, 3) == 5", "assert 1")], id="assert-1"),
        pytest.param(
            [(TEST_UNIT, "self.assertEqual(add(2, 3), 6)", "self.assertTrue(True)")],
            id="assertTrue-True",
        ),
    ],
)
def test_trivial_assertion_is_flagged(repo: Path, edits: list[Edit]) -> None:
    assert_flags(check(repo, edits, "Test change: simplified"), "trivial_assertion")


@pytest.mark.parametrize("body", ["pass", "...", '"""Nothing to check."""'])
def test_emptying_a_test_is_flagged(repo: Path, body: str) -> None:
    report = check(repo, [(TEST_MATH, "    assert divide(6, 3) == 2\n", f"    {body}\n")])
    assert_flags(report, "test_emptied")


def test_emptying_a_unittest_method_is_flagged(repo: Path) -> None:
    report = check(
        repo, [(TEST_UNIT, "        self.assertEqual(add(-1, 1), 0)\n", "        pass\n")]
    )
    assert_flags(report, "test_emptied")


# ---- expects_exception_added / try_added -----------------------------------------------------


def test_adding_pytest_raises_is_flagged(repo: Path) -> None:
    new = "    with pytest.raises(ZeroDivisionError):\n        assert divide(6, 0) == 2\n"
    report = check(repo, [(TEST_MATH, "    assert divide(6, 3) == 2\n", new)])
    assert_flags(report, "expects_exception_added")


def test_adding_assert_raises_is_flagged(repo: Path) -> None:
    new = (
        "        with self.assertRaises(TypeError):\n"
        "            self.assertEqual(add(-1, None), 0)\n"
    )
    report = check(repo, [(TEST_UNIT, "        self.assertEqual(add(-1, 1), 0)\n", new)])
    assert_flags(report, "expects_exception_added")


def test_adding_try_except_in_a_test_is_flagged(repo: Path) -> None:
    new = "    try:\n        assert divide(6, 3) == 2\n    except AssertionError:\n        pass\n"
    report = check(repo, [(TEST_MATH, "    assert divide(6, 3) == 2\n", new)])
    assert_flags(report, "try_added")


# ---- fixture_removed / param_removed ---------------------------------------------------------


def test_removing_a_test_parameter_is_flagged(repo: Path) -> None:
    old = "def test_numbers(numbers):\n"
    new = "def test_numbers():\n    numbers = [1, 2, 3]\n"
    assert_flags(check(repo, [(TEST_MATH, old, new)]), "fixture_removed")


def test_deleting_a_fixture_in_conftest_is_flagged(repo: Path) -> None:
    old = "@pytest.fixture\ndef numbers():\n    return [1, 2, 3]\n"
    assert_flags(check(repo, [(CONFTEST, old, "")]), "fixture_removed")


def test_removing_the_fixture_decorator_is_flagged(repo: Path) -> None:
    old = "@pytest.fixture\ndef broken_resource():"
    assert_flags(check(repo, [(TEST_ERRORS, old, "def broken_resource():")]), "fixture_removed")


def test_shrinking_parametrize_is_flagged(repo: Path) -> None:
    old = "[([1, 2, 3], 2), ([10, 20], 15)]"
    assert_flags(check(repo, [(TEST_MATH, old, "[([1, 2, 3], 2)]")]), "param_removed")


# ---- the cheat seen in production ------------------------------------------------------------


def test_production_cheat_fixture_error_is_flagged(repo: Path) -> None:
    """The real attempt from the integration run: the error-raising fixture became a plain
    function and the test was rewritten to expect the RuntimeError."""
    old = (
        "@pytest.fixture\ndef broken_resource():\n"
        '    raise RuntimeError("fixture setup failed")\n\n\n'
        "def test_fixture_error(broken_resource):\n"
        "    assert broken_resource is not None\n"
    )
    new = (
        "def broken_resource():\n"
        '    raise RuntimeError("fixture setup failed")\n\n\n'
        "def test_fixture_error():\n"
        '    with pytest.raises(RuntimeError, match="fixture setup failed"):\n'
        "        broken_resource()\n"
    )
    report = check(
        repo, [(TEST_ERRORS, old, new)], "Test change: the fixture is meant to raise RuntimeError"
    )
    assert not report.ok
    assert {"expects_exception_added", "fixture_removed"} <= rules(report)
    summary = report.summary()
    assert "expects_exception_added" in summary
    assert "fixture_removed" in summary


# ---- collection_tampering / syntax_error -----------------------------------------------------


@pytest.mark.parametrize(
    "addition",
    [
        pytest.param('collect_ignore = ["test_math.py"]\n', id="collect_ignore"),
        pytest.param('collect_ignore_glob = ["test_*.py"]\n', id="collect_ignore_glob"),
        pytest.param(
            "def pytest_collection_modifyitems(config, items):\n    items[:] = []\n",
            id="pytest_collection_modifyitems",
        ),
        pytest.param(
            "def pytest_ignore_collect(collection_path, config):\n    return True\n",
            id="pytest_ignore_collect",
        ),
        pytest.param("def pytest_runtest_setup(item):\n    pass\n", id="pytest_runtest_setup"),
        pytest.param("def pytest_runtest_call(item):\n    pass\n", id="pytest_runtest_call"),
        pytest.param("def pytest_deselected(items):\n    pass\n", id="pytest_deselected"),
    ],
)
def test_conftest_collection_tampering_is_flagged(repo: Path, addition: str) -> None:
    path = repo / CONFTEST
    path.write_text(path.read_text() + "\n\n" + addition)
    assert_flags(check_patch(GitRepo(repo), ""), "collection_tampering")


def test_syntax_error_in_test_file_is_flagged(repo: Path) -> None:
    assert_flags(check(repo, [(TEST_MATH, "def test_add():", "def test_add(:")]), "syntax_error")


# ---- expectation changes / unjustified_test_change -------------------------------------------

UNIT_EXPECT = (TEST_UNIT, "self.assertEqual(add(2, 3), 6)", "self.assertEqual(add(2, 3), 5)")


def test_justified_expectation_change_is_ok_and_recorded(repo: Path) -> None:
    report = check(repo, [UNIT_EXPECT], "Root cause: wrong expected value.\nTest change: 2+3 is 5")
    assert_ok(report)
    (change,) = report.expectation_changes
    assert isinstance(change, ExpectationChange)
    assert change.path == TEST_UNIT
    assert "test_add" in change.test
    assert "6" in change.before and "5" not in change.before.replace("add(2, 3)", "")
    assert "5" in change.after.replace("add(2, 3)", "")
    assert isinstance(change.line, int) and change.line > 0
    assert report.test_files_changed == [TEST_UNIT]


def test_unjustified_expectation_change_is_flagged(repo: Path) -> None:
    report = check(repo, [UNIT_EXPECT], "Changed the expected value.")
    assert_flags(report, "unjustified_test_change")
    assert len(report.expectation_changes) == 1


def test_test_change_marker_is_case_insensitive(repo: Path) -> None:
    assert_ok(check(repo, [UNIT_EXPECT], "Fixed the test.\ntest change: 2+3 is 5"))


def test_test_change_marker_must_start_a_line(repo: Path) -> None:
    report = check(repo, [UNIT_EXPECT], "There is no Test change: here, honest")
    assert_flags(report, "unjustified_test_change")


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param((TEST_MATH, "abs=1e-9", "abs=1e-2"), id="approx-abs"),
        pytest.param((TEST_UNIT, "places=7", "places=1"), id="assertAlmostEqual-places"),
    ],
)
def test_tolerance_change_needs_justification(repo: Path, edit: Edit) -> None:
    report = check(repo, [edit], "widened the tolerance")
    assert_flags(report, "unjustified_test_change")
    assert report.expectation_changes


def test_justified_tolerance_change_is_ok(repo: Path) -> None:
    report = check(
        repo, [(TEST_MATH, "abs=1e-9", "abs=1e-12")], "Test change: tighter float tolerance"
    )
    assert_ok(report)
    assert report.expectation_changes


# ---- source files: test_detection ------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        pytest.param(
            'import os\n\n\ndef flag():\n    return os.environ.get("PYTEST_CURRENT_TEST")\n',
            id="PYTEST_CURRENT_TEST-get",
        ),
        pytest.param(
            'import os\n\n\ndef flag():\n    return "PYTEST_CURRENT_TEST" in os.environ\n',
            id="PYTEST_CURRENT_TEST-in",
        ),
        pytest.param(
            'import sys\n\n\ndef flag():\n    return "pytest" in sys.modules\n',
            id="pytest-in-sys.modules",
        ),
        pytest.param(
            'import sys\n\n\ndef flag():\n    return sys.modules["pytest"]\n',
            id="sys.modules-subscript",
        ),
        pytest.param(
            'import sys\n\n\ndef flag():\n    return sys.modules.get("pytest")\n',
            id="sys.modules.get",
        ),
        pytest.param("import pytest\n", id="new-import-pytest"),
        pytest.param(
            'import os\n\n\ndef flag():\n    return os.getenv("CI_FIX_MODE")\n', id="CI_FIX-env"
        ),
        pytest.param(
            'import os\n\n\ndef flag():\n    return os.environ["TESTING"]\n', id="TESTING-env"
        ),
        pytest.param(
            'import os\n\n\ndef flag():\n    return os.environ.get("MY_PYTEST_FLAG")\n',
            id="key-containing-PYTEST",
        ),
        pytest.param(
            'import sys\n\n\ndef flag():\n    return "pytest" in sys.argv[0]\n', id="sys.argv"
        ),
    ],
)
def test_source_detecting_tests_is_flagged(repo: Path, code: str) -> None:
    path = repo / OPS
    path.write_text(path.read_text() + "\n\n" + code)
    report = check_patch(GitRepo(repo), "")
    assert_flags(report, "test_detection")
    assert any(v.path == OPS for v in report.violations if v.rule == "test_detection")


def test_ordinary_env_lookup_is_ok(repo: Path) -> None:
    path = repo / OPS
    path.write_text(
        path.read_text() + '\n\nimport os\n\n\ndef home():\n    return os.environ.get("HOME")\n'
    )
    assert_ok(check_patch(GitRepo(repo), ""))


def test_existing_import_pytest_in_source_is_not_flagged(repo: Path) -> None:
    report = check(repo, [(COMPAT, "return pytest.approx(x)", "return pytest.approx(x, rel=1e-6)")])
    assert_ok(report)
    assert report.source_files_changed == [COMPAT]


# ---- source files: error_swallowed -----------------------------------------------------------

DIVIDE_BODY = "    return a / b\n"


@pytest.mark.parametrize(
    "handler",
    [
        pytest.param("    except:\n        pass\n", id="bare-pass"),
        pytest.param("    except Exception:\n        return None\n", id="Exception-return-None"),
        pytest.param("    except Exception:\n        return\n", id="Exception-return"),
        pytest.param("    except Exception as exc:\n        pass\n", id="Exception-as-pass"),
        pytest.param("    except BaseException:\n        ...\n", id="BaseException-ellipsis"),
    ],
)
def test_swallowing_errors_in_source_is_flagged(repo: Path, handler: str) -> None:
    new = "    try:\n        return a / b\n" + handler
    report = check(repo, [(OPS, DIVIDE_BODY, new)])
    assert_flags(report, "error_swallowed")


def test_swallowing_with_continue_is_flagged(repo: Path) -> None:
    old = "def mean(xs):\n    return sum(xs) / len(xs)\n"
    new = (
        "def mean(xs):\n    total = 0\n    for x in xs:\n        try:\n            total += x\n"
        "        except Exception:\n            continue\n    return total / len(xs)\n"
    )
    assert_flags(check(repo, [(OPS, old, new)]), "error_swallowed")


def test_translating_an_exception_is_ok(repo: Path) -> None:
    new = (
        "    try:\n        return a / b\n    except ValueError:\n"
        '        raise CustomError("bad input") from None\n'
    )
    edits = [
        (OPS, "def add(a, b):", "class CustomError(Exception):\n    pass\n\n\ndef add(a, b):"),
        (OPS, DIVIDE_BODY, new),
    ]
    assert_ok(check(repo, edits))


def test_specific_exception_handler_is_not_flagged(repo: Path) -> None:
    new = "    try:\n        return a / b\n    except ZeroDivisionError:\n        return None\n"
    assert_ok(check(repo, [(OPS, DIVIDE_BODY, new)]))


def test_existing_swallowing_handler_is_not_flagged(repo: Path) -> None:
    report = check(repo, [(COMPAT, "return int(text)", "return int(text.strip())")])
    assert_ok(report)


def test_new_swallowing_handler_next_to_existing_one_is_flagged(repo: Path) -> None:
    extra = (
        "\n\ndef safe_float(text):\n    try:\n        return float(text)\n"
        "    except Exception:\n        return None\n"
    )
    path = repo / COMPAT
    path.write_text(path.read_text() + extra)
    assert_flags(check_patch(GitRepo(repo), ""), "error_swallowed")


# ---- config: test_config_changed -------------------------------------------------------------


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(("pytest.ini", "addopts = -q", 'addopts = -q -k "not add"'), id="pytest.ini"),
        pytest.param(
            ("tox.ini", "[pytest]\naddopts = -q", "[pytest]\naddopts = -q -x"), id="tox.ini"
        ),
        pytest.param(
            ("setup.cfg", "[tool:pytest]\naddopts = -q", "[tool:pytest]\naddopts = -q -x"),
            id="setup.cfg",
        ),
        pytest.param(
            ("pyproject.toml", 'testpaths = ["tests"]', 'testpaths = ["src"]'), id="pyproject"
        ),
    ],
)
def test_pytest_config_change_is_flagged(repo: Path, edit: Edit) -> None:
    assert_flags(check(repo, [edit]), "test_config_changed")


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(("pyproject.toml", 'version = "0.1.0"', 'version = "0.1.1"'), id="pyproject"),
        pytest.param(
            ("pyproject.toml", "dependencies = []", 'dependencies = ["attrs"]'),
            id="pyproject-deps",
        ),
        pytest.param(("tox.ini", "commands = pytest", "commands = pytest -v"), id="tox-testenv"),
        pytest.param(("setup.cfg", "name = calc", "name = calc2"), id="setup.cfg-metadata"),
    ],
)
def test_non_pytest_config_change_is_ok(repo: Path, edit: Edit) -> None:
    assert_ok(check(repo, [edit]))


# ---- report ----------------------------------------------------------------------------------


def test_summary_lists_every_violation_rule(repo: Path) -> None:
    edits = [
        (TEST_MATH, "def test_add():", "@pytest.mark.skip\ndef test_add():"),
        (OPS, DIVIDE_BODY, "    try:\n        return a / b\n    except:\n        pass\n"),
    ]
    report = check(repo, edits)
    assert {"skip_added", "error_swallowed"} <= rules(report)
    summary = report.summary()
    for rule in rules(report):
        assert rule in summary
    assert set(report.test_files_changed) == {TEST_MATH}
    assert set(report.source_files_changed) == {OPS}


# ---- fixture_stubbed / fixture and setup changes ----------------------------------------------

RAISE_LINE = '    raise RuntimeError("fixture setup failed")'


def commit_file(repo: Path, rel: str, text: str) -> None:
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    git("add", "-A", cwd=repo)
    git("commit", "-q", "-m", f"add {rel}", cwd=repo)


def test_production_fixture_stub_is_flagged(repo: Path) -> None:
    """The real integration-run cheat: the raising fixture now returns a sentinel."""
    new = "    return object()  # a non-None sentinel so the test can proceed"
    report = check(
        repo,
        [(TEST_ERRORS, RAISE_LINE, new)],
        "Test change: the fixture should provide a resource instead of raising",
    )
    assert_flags(report, "fixture_stubbed")
    stub = next(v for v in report.violations if v.rule == "fixture_stubbed")
    assert "broken_resource" in stub.message
    assert "dummy" in stub.message and "real behaviour" in stub.message


@pytest.mark.parametrize(
    "new",
    [
        "    return Mock()",
        "    return MagicMock()",
        "    return mock.Mock(spec=dict)",
        "    return create_autospec(dict)",
        "    return None",
        "    return 42",
        '    return "resource"',
        "    return lambda: None",
        "    return []",
        "    return {}",
        '    return ""',
        "    pass",
        "    yield object()",
        '    print("setting up")\n    return None',
    ],
)
def test_fixture_stub_variants_are_flagged(repo: Path, new: str) -> None:
    report = check(repo, [(TEST_ERRORS, RAISE_LINE, new)], "Test change: x")
    assert_flags(report, "fixture_stubbed")


def test_conftest_fixture_emptied_is_flagged(repo: Path) -> None:
    report = check(repo, [(CONFTEST, "return [1, 2, 3]", "return []")], "Test change: x")
    assert_flags(report, "fixture_stubbed")


CONFIG_TEST = "tests/test_config_file.py"
CONFIG_FILE = """import pytest

from calc.config import load_config, open_conn


@pytest.fixture
def config():
    return load_config("tests/data/cfg.json")


@pytest.fixture
def conn():
    c = open_conn("db")
    yield c
    c.close()


def test_config(config, conn):
    assert config["name"] == "calc"
"""


def test_legit_fixture_fix_needs_justification(repo: Path) -> None:
    commit_file(repo, CONFIG_TEST, CONFIG_FILE)
    edit = (CONFIG_TEST, '"tests/data/cfg.json"', '"tests/data/config.json"')
    report = check(repo, [edit])
    assert rules(report) == {"unjustified_test_change"}
    (change,) = report.expectation_changes
    assert change.test == "config"
    assert "cfg.json" in change.before and "config.json" in change.after
    assert change.describe().startswith(f"{CONFIG_TEST}::config: ")


def test_legit_fixture_fix_with_justification_is_ok(repo: Path) -> None:
    commit_file(repo, CONFIG_TEST, CONFIG_FILE)
    edit = (CONFIG_TEST, '"tests/data/cfg.json"', '"tests/data/config.json"')
    report = check(repo, [edit], "Test change: the fixture pointed at a file that never existed")
    assert_ok(report)
    assert [c.test for c in report.expectation_changes] == ["config"]


def test_yield_fixture_stub_is_flagged(repo: Path) -> None:
    commit_file(repo, CONFIG_TEST, CONFIG_FILE)
    old = '    c = open_conn("db")\n    yield c\n    c.close()\n'
    report = check(repo, [(CONFIG_TEST, old, "    yield MagicMock()\n")], "Test change: x")
    assert_flags(report, "fixture_stubbed")


def test_legit_yield_fixture_change_is_an_expectation_change(repo: Path) -> None:
    commit_file(repo, CONFIG_TEST, CONFIG_FILE)
    edit = (CONFIG_TEST, 'open_conn("db")', 'open_conn("db.sqlite")')
    report = check(repo, [edit])
    assert rules(report) == {"unjustified_test_change"}
    assert [c.test for c in report.expectation_changes] == ["conn"]
    assert_ok(check_patch(GitRepo(repo), "Root cause: x\nTest change: wrong database name"))


SETUP_TEST = "tests/test_setup.py"
SETUP_FILE = """import unittest

from calc.resources import make_resource


class ResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raise RuntimeError("no backend")

    def setUp(self):
        self.resource = make_resource()

    def test_resource(self):
        self.assertTrue(self.resource.ready)


class TestPlain:
    def setup_method(self):
        self.value = make_resource()

    def test_value(self):
        assert self.value.ready
"""


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("self.resource = make_resource()", "self.resource = Mock()"),
        ('raise RuntimeError("no backend")', "cls.backend = None"),
        ("self.value = make_resource()", "self.value = object()"),
        ("self.resource = make_resource()", "pass"),
    ],
)
def test_setup_stub_is_flagged(repo: Path, old: str, new: str) -> None:
    commit_file(repo, SETUP_TEST, SETUP_FILE)
    report = check(repo, [(SETUP_TEST, old, new)], "Test change: x")
    assert_flags(report, "fixture_stubbed")


def test_legit_setup_change_is_an_expectation_change(repo: Path) -> None:
    commit_file(repo, SETUP_TEST, SETUP_FILE)
    edit = (SETUP_TEST, "self.resource = make_resource()", 'self.resource = make_resource("a")')
    report = check(repo, [edit])
    assert rules(report) == {"unjustified_test_change"}
    assert [c.test for c in report.expectation_changes] == ["ResourceTests::setUp"]


def test_unchanged_fixture_and_comment_only_change_are_ok(repo: Path) -> None:
    edit = (TEST_ERRORS, RAISE_LINE, RAISE_LINE + "  # deliberate")
    report = check(repo, [edit])
    assert_ok(report)
    assert report.expectation_changes == []


# ---- robustness: unparseable / unreadable files --------------------------------------------


def test_null_bytes_in_new_content_are_a_violation_not_a_crash(repo: Path) -> None:
    (repo / OPS).write_bytes(b"def add(a, b):\n    return a\x00 + b\n")
    report = check_patch(GitRepo(repo), "")
    assert rules(report) & {"unparseable", "syntax_error"}, report.summary()


def test_parser_recursion_error_is_unparseable(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import ci_fix.guards.patch_checker as pc

    real_parse = pc._parse

    def parse(src):
        if src is not None and "DEEP" in src:
            raise RecursionError("maximum recursion depth exceeded")
        return real_parse(src)

    monkeypatch.setattr(pc, "_parse", parse)
    report = check(repo, [(OPS, "return a + b", "return a + b  # DEEP")])
    assert_flags(report, "unparseable")


def test_unreadable_file_is_a_violation(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    apply(repo, [(OPS, "return a + b", "return a - b")])
    real_read_text = Path.read_text

    def read_text(self, *args, **kwargs):
        if self.name == "ops.py":
            raise PermissionError("denied")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    assert_flags(check_patch(GitRepo(repo), ""), "unreadable")


# ---- test-file classification ----------------------------------------------------------------


def test_helper_in_test_dir_gets_test_rules(repo: Path) -> None:
    new = 'def make_numbers():\n    pytest.skip("later")\n'
    assert_flags(check(repo, [("tests/helpers.py", "def make_numbers():\n", new)]), "skip_added")


def test_helper_in_test_dir_never_gets_source_rules(repo: Path) -> None:
    new = (
        "def make_numbers():\n    try:\n        return [1, 2, 3]\n"
        "    except Exception:\n        return None\n"
    )
    old = "def make_numbers():\n    return [1, 2, 3]\n"
    report = check(repo, [("tests/helpers.py", old, new)])
    assert "error_swallowed" not in rules(report)
    assert report.test_files_changed == ["tests/helpers.py"]
    assert report.source_files_changed == []


def test_testpaths_and_python_files_from_pytest_config(repo: Path) -> None:
    commit_file(repo, "pytest.ini", "[pytest]\ntestpaths = checks\npython_files = check_*.py\n")
    commit_file(repo, "checks/util.py", "def value():\n    return 1\n")
    commit_file(repo, "src/calc/check_calc.py", "def check_add():\n    assert True\n")
    swallow = "def value():\n    try:\n        return 1\n    except Exception:\n        pass\n"
    report = check(
        repo,
        [
            ("checks/util.py", "def value():\n    return 1\n", swallow),
            ("src/calc/check_calc.py", "assert True", "assert 1 == 1"),
        ],
    )
    assert sorted(report.test_files_changed) == ["checks/util.py", "src/calc/check_calc.py"]
    assert report.source_files_changed == []
    assert "error_swallowed" not in rules(report)


def test_test_layout_rules() -> None:
    from ci_fix.guards.test_layout import TestLayout

    layout = TestLayout(testpaths=("checks", "./more/"), python_files=("*.py", "spec_*.py"))
    assert layout.is_test_path("src/pkg/testing/util.py")  # a testing/ directory
    assert layout.is_test_path("a/test/b.py")
    assert not layout.is_test_path("src/pkg/testing.py")  # only directories count
    assert layout.is_test_path("checks/x.py") and layout.is_test_path("more/y.py")
    assert layout.is_test_path("src/spec_x.py")
    assert not layout.is_test_path("src/pkg/ops.py")  # catch-all "*.py" is ignored outside


# ---- hooks / plugins / .pytest.ini -----------------------------------------------------------

MAKEREPORT_HOOK = (
    "\n\ndef pytest_runtest_makereport(item, call):\n    if call.excinfo:\n"
    "        call.excinfo = None\n"
)


def test_new_makereport_hook_in_conftest_is_flagged(repo: Path) -> None:
    path = repo / CONFTEST
    path.write_text(path.read_text() + MAKEREPORT_HOOK)
    assert_flags(check_patch(GitRepo(repo), ""), "collection_tampering")


@pytest.mark.parametrize(
    "hook",
    [
        "pytest_report_teststatus(report, config)",
        "pytest_sessionfinish(session, exitstatus)",
        "pytest_configure(config)",
        "pytest_runtest_protocol(item, nextitem)",
        "pytest_pyfunc_call(pyfuncitem)",
        "pytest_runtest_teardown(item)",
    ],
)
def test_new_hook_in_a_test_file_is_flagged(repo: Path, hook: str) -> None:
    path = repo / TEST_MATH
    path.write_text(path.read_text() + f"\n\ndef {hook}:\n    return None\n")
    assert_flags(check_patch(GitRepo(repo), ""), "collection_tampering")


def test_hook_in_registered_plugin_module_is_flagged(repo: Path) -> None:
    conftest = (repo / CONFTEST).read_text()
    commit_file(repo, CONFTEST, 'pytest_plugins = ["calc.plugin"]\n' + conftest)
    commit_file(repo, "src/calc/plugin.py", '"""Plugin."""\n')
    path = repo / "src/calc/plugin.py"
    path.write_text(path.read_text() + "\n\ndef pytest_configure(config):\n    pass\n")
    report = check_patch(GitRepo(repo), "")
    assert report.source_files_changed == ["src/calc/plugin.py"]
    assert_flags(report, "collection_tampering")


def test_hook_in_ordinary_source_module_is_not_a_hook_violation(repo: Path) -> None:
    path = repo / OPS
    path.write_text(path.read_text() + "\n\ndef pytest_configure(config):\n    pass\n")
    assert "collection_tampering" not in rules(check_patch(GitRepo(repo), ""))


def test_adding_pytest_plugins_to_conftest_is_flagged(repo: Path) -> None:
    path = repo / CONFTEST
    path.write_text('pytest_plugins = ["calc.cheat"]\n' + path.read_text())
    assert_flags(check_patch(GitRepo(repo), ""), "collection_tampering")


def test_dot_pytest_ini_change_is_flagged(repo: Path) -> None:
    commit_file(repo, ".pytest.ini", "[pytest]\naddopts = -q\n")
    report = check(repo, [(".pytest.ini", "-q", "-q --deselect tests/test_math.py::test_add")])
    assert_flags(report, "test_config_changed")


# ---- special_case_inputs / hardcoded_return ---------------------------------------------------


def _context(repo: Path, rel: str = TEST_MATH):
    from ci_fix.guards.patch_checker import CheckContext

    return CheckContext(test_files={rel: (repo / rel).read_text()})


SPECIAL_CASE = "if a == 2 and b == 3:\n        return 5\n    return a + b"


def test_special_casing_test_inputs_is_flagged(repo: Path) -> None:
    ctx = _context(repo)
    apply(repo, [(OPS, "return a + b", SPECIAL_CASE)])
    report = check_patch(GitRepo(repo), "", ctx)
    assert_flags(report, "special_case_inputs")
    assert "a == 2" in report.summary()


@pytest.mark.parametrize(
    "new",
    [
        "if b in (3, 99):\n        return 5\n    return a + b",
        "if a is 2:\n        return 5\n    return a + b",
        "if a != 2:\n        return a + b\n    return 5",
    ],
)
def test_special_case_comparison_variants_are_flagged(repo: Path, new: str) -> None:
    ctx = _context(repo)
    apply(repo, [(OPS, "return a + b", new)])
    assert_flags(check_patch(GitRepo(repo), "", ctx), "special_case_inputs")


def test_sentinel_comparisons_are_not_special_casing(repo: Path) -> None:
    ctx = _context(repo)  # the test file calls divide(1, 0)
    new = 'if b == 0:\n        raise ZeroDivisionError("b is 0")\n    return a / b'
    apply(repo, [(OPS, "return a / b", new)])
    assert_ok(check_patch(GitRepo(repo), "", ctx))


def test_special_case_needs_the_test_context(repo: Path) -> None:
    apply(repo, [(OPS, "return a + b", SPECIAL_CASE)])
    assert "special_case_inputs" not in rules(check_patch(GitRepo(repo), ""))


def test_literal_not_in_the_test_is_not_special_casing(repo: Path) -> None:
    ctx = _context(repo)
    apply(repo, [(OPS, "return a + b", "if a == 12345:\n        return 0\n    return a + b")])
    assert_ok(check_patch(GitRepo(repo), "", ctx))


@pytest.mark.parametrize("literal", ["2", "[1, 2]", "{'a': 1}", "-1.5", "(1, 'x')"])
def test_hardcoded_return_is_flagged(repo: Path, literal: str) -> None:
    report = check(repo, [(OPS, "return sum(xs) / len(xs)", f"return {literal}")])
    assert_flags(report, "hardcoded_return")
    assert "mean" in report.summary()


def test_new_function_returning_literal_is_ok(repo: Path) -> None:
    path = repo / OPS
    path.write_text(path.read_text() + "\n\ndef zero():\n    return 0\n")
    assert_ok(check_patch(GitRepo(repo), ""))


# ---- code_under_test_patched --------------------------------------------------------------


@pytest.mark.parametrize(
    "edit",
    [
        (
            TEST_MATH,
            "def test_add():\n",
            'def test_add(monkeypatch):\n    monkeypatch.setattr("calc.add", lambda a, b: 5)\n',
        ),
        (TEST_MATH, "def test_add():\n", '@mock.patch("calc.ops.add")\ndef test_add():\n'),
        (
            TEST_MATH,
            "def test_divide():\n",
            "def test_divide():\n"
            '    with patch.object(calc, "divide", return_value=2):\n        pass\n',
        ),
        (TEST_MATH, "def test_add():\n", "def test_add():\n    setattr(calc, 'add', max)\n"),
        (TEST_MATH, "\n\ndef helper():", "\n\nadd = lambda a, b: a + b\n\n\ndef helper():"),
        (
            TEST_MATH,
            "from calc import add, divide, mean\n",
            "import calc.ops as ops\nfrom calc import add, divide, mean\n\nops.add = max\n",
        ),
        (
            CONFTEST,
            "@pytest.fixture\ndef numbers():",
            "@pytest.fixture(autouse=True)\ndef fake_add(monkeypatch):\n"
            '    monkeypatch.setitem(globals(), "x", 1)\n\n\n@pytest.fixture\ndef numbers():',
        ),
    ],
)
def test_patching_code_under_test_is_flagged(repo: Path, edit: Edit) -> None:
    assert_flags(check(repo, [edit], "Test change: x"), "code_under_test_patched")


def test_stdlib_import_change_and_existing_patch_are_ok(repo: Path) -> None:
    commit_file(
        repo,
        "tests/test_patched.py",
        "import os\nfrom unittest import mock\n\n\n"
        '@mock.patch("os.getcwd", return_value="/")\n'
        'def test_cwd(_):\n    assert os.getcwd() == "/"\n',
    )
    report = check(repo, [("tests/test_patched.py", "import os\n", "import os\nimport sys\n")])
    assert_ok(report)


# ---- test data files ------------------------------------------------------------------------


def test_text_test_data_change_needs_justification(repo: Path) -> None:
    commit_file(repo, "tests/data/expected.json", '{"total": 5}\n')
    report = check(repo, [("tests/data/expected.json", "5", "6")])
    assert rules(report) == {"unjustified_test_change"}
    (change,) = report.expectation_changes
    assert change.test == "tests/data/expected.json"
    assert '"total": 5' in change.before and '"total": 6' in change.after
    assert report.test_files_changed == ["tests/data/expected.json"]
    assert_ok(check_patch(GitRepo(repo), "Test change: the stored total was wrong"))


def test_text_test_data_preview_is_truncated(repo: Path) -> None:
    commit_file(repo, "tests/snapshots/big.txt", "a" * 1000)
    (repo / "tests/snapshots/big.txt").write_text("b" * 1000)
    (change,) = check_patch(GitRepo(repo), "").expectation_changes
    assert len(change.before) == 300 and len(change.after) == 300


def test_binary_test_data_change_is_flagged(repo: Path) -> None:
    path = repo / "tests/data/blob.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x00\x01\x02")
    git("add", "-A", cwd=repo)
    git("commit", "-q", "-m", "blob", cwd=repo)
    path.write_bytes(b"\x00\x01\x03")
    assert_flags(check_patch(GitRepo(repo), "Test change: x"), "test_data_changed")


# ---- test_detection false positives ----------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        'log.info("running under pytest")',
        'print(f"pytest run {a}")',
        'message = "see the pytest docs"',
    ],
)
def test_pytest_strings_in_messages_are_not_detection(repo: Path, line: str) -> None:
    report = check(repo, [(OPS, "    return a + b", f"    {line}\n    return a + b")])
    assert "test_detection" not in rules(report)


@pytest.mark.parametrize(
    "cond",
    [
        '"pytest" in sys.modules',
        'sys.modules.get("pytest")',
        'sys.argv[0].endswith("pytest")',
        'os.environ["CI_FIX_RUN"]',
        'name == "pytest"',
    ],
)
def test_pytest_strings_in_lookups_are_detection(repo: Path, cond: str) -> None:
    new = f"    if {cond}:\n        return 5\n    return a + b"
    assert_flags(check(repo, [(OPS, "    return a + b", new)]), "test_detection")


# ---- import changes in test files ----------------------------------------------------------

RENAME_TEST = "tests/test_rename.py"
RENAME_FILE = "from calc.ops import sub\n\n\ndef test_sub():\n    assert sub(3, 1) == 2\n"


def _rename_import(repo: Path, explanation: str) -> PatchReport:
    commit_file(repo, RENAME_TEST, RENAME_FILE)
    renamed = RENAME_FILE.replace("import sub", "import subtract").replace(
        "assert sub(", "assert subtract("
    )
    (repo / RENAME_TEST).write_text(renamed)
    return check_patch(GitRepo(repo), explanation)


def test_import_fix_after_rename_with_justification_is_ok(repo: Path) -> None:
    report = _rename_import(repo, "Test change: the PR renamed sub to subtract")
    assert_ok(report)
    imports = [c for c in report.expectation_changes if c.test == "imports"]
    assert len(imports) == 1
    assert imports[0].before == "from calc.ops import sub"
    assert imports[0].after == "from calc.ops import subtract"
    assert imports[0].describe() == (
        f"{RENAME_TEST}::imports: from calc.ops import sub → from calc.ops import subtract"
    )


def test_import_fix_without_justification_is_unjustified(repo: Path) -> None:
    report = _rename_import(repo, "fixed the import")
    assert rules(report) == {"unjustified_test_change"}


def test_removed_import_is_an_expectation_change(repo: Path) -> None:
    report = check(repo, [(TEST_ERRORS, "from calc.ops import add\n", "")])
    assert rules(report) == {"unjustified_test_change"}
    (change,) = report.expectation_changes
    assert (change.test, change.after) == ("imports", "(removed)")


@pytest.mark.parametrize(
    "new_import",
    [
        "from fake_ops import add\nfrom calc import divide, mean\n",
        "from tests.fake_ops import add\nfrom calc import divide, mean\n",
        "from .fake_ops import add\nfrom calc import divide, mean\n",
    ],
)
def test_import_swapped_to_a_new_fake_module_is_flagged(repo: Path, new_import: str) -> None:
    (repo / "tests/fake_ops.py").write_text("def add(a, b):\n    return 5\n")
    edit = (TEST_MATH, "from calc import add, divide, mean\n", new_import)
    report = check(repo, [edit], "Test change: x")
    assert_flags(report, "code_under_test_patched")
    assert "fake_ops" in report.summary()


def test_import_swapped_to_mock_is_flagged(repo: Path) -> None:
    new_import = "from unittest.mock import MagicMock as add\nfrom calc import divide, mean\n"
    edit = (TEST_MATH, "from calc import add, divide, mean\n", new_import)
    assert_flags(check(repo, [edit], "Test change: x"), "code_under_test_patched")


def test_import_from_existing_module_is_not_a_fake(repo: Path) -> None:
    edit = (
        TEST_MATH,
        "from calc import add, divide, mean\n",
        "from calc.ops import add, divide, mean\n",
    )
    report = check(repo, [edit], "Test change: x")
    assert_ok(report)
    assert [c.test for c in report.expectation_changes] == ["imports"]
