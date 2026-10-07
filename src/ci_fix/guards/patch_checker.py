"""Deterministic integrity checks on a proposed fix (stdlib ``ast`` only).

:func:`check_patch` compares every changed ``.py`` file (and pytest config file) between
``HEAD`` and the working tree and reports rule violations. A patch with any violation is
rejected and the :meth:`PatchReport.summary` is fed back to the fixer.

Test files are test modules (``python_files`` from the pytest config at HEAD, default
``test_*.py``/``*_test.py``, and ``conftest.py``) and every ``.py`` file in a test directory
(a ``tests``/``test``/``testing`` directory or a configured ``testpaths`` entry); see
:mod:`ci_fix.guards.test_layout`. Test files never get source rules; every other ``.py`` file
is source. Tests are module-level ``test*`` functions and ``test*`` methods of ``Test*`` classes
(or classes subclassing ``*TestCase``), keyed ``name`` or ``Class::name``.

Rule ids (referenced in PR descriptions):

Any ``.py`` file
- ``syntax_error`` — the new content does not parse.
- ``unparseable`` — the new content is too deeply nested or otherwise can't be parsed.
- ``unreadable`` — the changed file can't be read.

Test files
- ``test_removed`` — a test or test class that existed in HEAD is gone (deleted or renamed,
  including a deleted test file).
- ``skip_added`` — a new skip/xfail: ``pytest.mark.skip/skipif/xfail`` and
  ``unittest.skip/skipIf/skipUnless/expectedFailure`` (decorators, ``pytestmark``,
  ``pytest.param(marks=...)``) or calls to ``pytest.skip/xfail/importorskip``/``self.skipTest``.
- ``assertion_removed`` — a test has fewer assertions than before. Counted: ``assert``
  statements, calls named ``assert*`` (``self.assertEqual``, ``mock.assert_called_once``, …)
  and ``pytest.raises``/``pytest.warns``/``pytest.deprecated_call``.
- ``trivial_assertion`` — a new always-true assertion (``assert True``, ``assert 1``,
  ``assert x == x``, ``self.assertTrue(True)``, …).
- ``test_emptied`` — a test body is now only ``pass``/``...``/a docstring.
- ``expects_exception_added`` — ``pytest.raises``/``assertRaises*`` added to a test that had none
  (rewriting the test to expect the error).
- ``try_added`` — a ``try`` (or ``contextlib.suppress``) added to a test that had none.
- ``fixture_removed`` — a test lost parameters (fixtures), or a ``@pytest.fixture`` was deleted.
- ``param_removed`` — fewer literal ``pytest.mark.parametrize`` cases than before.
- ``fixture_stubbed`` — an existing fixture (``@pytest.fixture``) or setup/teardown method
  (``setUp``/``setUpClass``/``tearDown``…, ``setup_method``/``setup_function``/
  ``setup_module``…) whose body now only returns/yields/assigns a dummy stand-in (a
  constant, ``None``, ``object()``, ``Mock()``/``MagicMock()``/``create_autospec(...)``, an
  empty literal, a lambda), or whose ``raise`` statements were all replaced by returning
  such a stand-in — the test then passes without exercising real behaviour.
- ``collection_tampering`` — in any test file, or a module registered as a plugin via
  ``pytest_plugins``: new ``collect_ignore``/``collect_ignore_glob``/``pytest_plugins``
  entries, or added/changed pytest hooks that can hide failures (see ``TAMPERING_HOOKS``:
  collection, ``pytest_runtest_*``, ``pytest_runtest_makereport``,
  ``pytest_report_teststatus``, ``pytest_sessionfinish``, ``pytest_configure``,
  ``pytest_pyfunc_call``, …).
- ``code_under_test_patched`` — new patching or substitution in a test file (new fixtures
  included): ``monkeypatch.setattr/setitem/delattr/delitem``, builtin ``setattr``/``delattr``,
  ``mock.patch(...)``/``patch.object(...)`` calls or decorators, rebinding an imported name
  or an attribute of an imported module (``ops.add = ...``, ``add = lambda ...``), or
  importing a name from ``mock``/``unittest.mock`` or from a ``.py`` file this patch
  created (a fake). Other changed/removed imports of non-stdlib code are expectation changes
  (test ``imports``), e.g. fixing an import after the PR renamed a function.
- ``test_data_changed`` — a binary file in a test directory changed. (A changed text file
  there — data, snapshots, JSON/YAML — is an expectation change.)
- ``unjustified_test_change`` — test expectations changed (see below) but the fixer's
  explanation has no ``Test change: <why the old expectation was wrong>`` line.

Expectation changes are not violations by themselves; they are recorded in
:attr:`PatchReport.expectation_changes` (for the PR description) and must be justified. They
are: a changed assertion, any other change to an existing fixture or setup/teardown
method (before/after = the function source), a changed tolerance argument (``abs=``,
``rel=``, ``places=``, ``delta=``, ``atol=``, ``rtol=``), a change to the literal values
assigned to a variable an assertion uses (``expected = 5`` → ``6``; renaming is fine), and
a change to the literal values of a module-level assignment in a test file.

Source files
- ``test_detection`` — new code that detects tests: a string mentioning ``pytest`` or
  ``CI_FIX`` used in a comparison, an ``in`` test, a subscript (``sys.modules["pytest"]``,
  ``os.environ[...]``) or a lookup call (``.get``, ``getenv``, ``import_module``,
  ``endswith``, …) — not in log messages, f-strings or docstrings; ``import pytest``/
  ``from pytest ...``; names containing ``PYTEST``; environment lookups of ``TESTING``.
- ``special_case_inputs`` — a new ``==``/``!=``/``in``/``is`` comparison between a name and
  a literal that also appears in the failing test's file (``if a == 2 and b == 3``).
  Common sentinels (``None``, ``True``, ``False``, ``0``, ``1``, ``-1``, ``""``) are ignored.
- ``hardcoded_return`` — an existing function whose body is now only ``return <literal>``
  (or a literal container) where before it computed something.
- ``error_swallowed`` — a new bare/``Exception``/``BaseException`` handler whose body only
  passes, continues or returns a constant.

Config files
- ``test_config_changed`` — ``pytest.ini``/``.pytest.ini`` changed, or the pytest section
  of ``tox.ini`` (``[pytest]``), ``setup.cfg`` (``[tool:pytest]``) or ``pyproject.toml``
  (``[tool.pytest]``, incl. ``ini_options``) changed. Other changes to these files are allowed.
"""

from __future__ import annotations

import ast
import configparser
import difflib
import re
import sys
import tomllib
from collections import Counter
from collections.abc import Hashable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from pydantic import BaseModel, Field

from ci_fix.guards.test_layout import (
    DEFAULT_LAYOUT,
    TestLayout,
    head_plugin_paths,
    plugin_paths,
    pytest_plugins_in,
    read_layout,
)
from ci_fix.logging_setup import get_logger
from ci_fix.tools.git import GitRepo

log = get_logger(__name__)

PYTEST_INIS = frozenset({"pytest.ini", ".pytest.ini"})
CONFIG_FILES = PYTEST_INIS | {"tox.ini", "setup.cfg", "pyproject.toml"}
_PARSE_ERRORS = (SyntaxError, ValueError, RecursionError, MemoryError)
DATA_PREVIEW_CHARS = 300
INI_PYTEST_SECTIONS = ("pytest", "tool:pytest")
SKIP_NAMES = frozenset(
    {
        "skip",
        "skipif",
        "xfail",
        "skipIf",
        "skipUnless",
        "expectedFailure",
        "skipTest",
        "importorskip",
    }
)
RAISES_NAMES = frozenset({"raises", "assertRaises", "assertRaisesRegex", "assertRaisesRegexp"})
EXPECT_CALL_NAMES = frozenset({"raises", "warns", "deprecated_call"})
TOLERANCE_KWARGS = frozenset({"abs", "rel", "places", "delta", "atol", "rtol"})
SETUP_NAMES = frozenset(
    {
        "setUp",
        "setUpClass",
        "setUpModule",
        "tearDown",
        "tearDownClass",
        "tearDownModule",
        "setup_method",
        "setup_function",
        "setup_module",
        "setup_class",
        "teardown_method",
        "teardown_function",
        "teardown_module",
        "teardown_class",
    }
)
# Calls that build a dummy stand-in instead of the real resource.
STAND_IN_CALLS = frozenset(
    {
        "object",
        "Mock",
        "MagicMock",
        "AsyncMock",
        "NonCallableMock",
        "NonCallableMagicMock",
        "PropertyMock",
        "create_autospec",
    }
)
EMPTY_CALLS = frozenset({"dict", "list", "set", "tuple", "str", "bytes", "frozenset"})
STUBBED_MESSAGE = (
    "fixture replaced with a dummy so the test passes without exercising real behaviour"
)
COLLECT_IGNORE_NAMES = frozenset({"collect_ignore", "collect_ignore_glob", "pytest_plugins"})
# Hooks that can deselect, skip, rewrite or hide test results.
TAMPERING_HOOKS = frozenset(
    {
        "pytest_collection_modifyitems",
        "pytest_ignore_collect",
        "pytest_deselected",
        "pytest_runtest_setup",
        "pytest_runtest_call",
        "pytest_runtest_teardown",
        "pytest_runtest_protocol",
        "pytest_runtest_makereport",
        "pytest_report_teststatus",
        "pytest_pyfunc_call",
        "pytest_sessionfinish",
        "pytest_configure",
    }
)
PATCH_METHODS = frozenset({"setattr", "setitem", "delattr", "delitem"})
# Literals too common to signal special-casing a test input.
SENTINEL_LITERALS = frozenset(
    {("NoneType", None), ("bool", True), ("bool", False), ("int", 0), ("int", 1), ("int", -1)}
    | {("str", ""), ("float", 0.0)}
)
# Calls whose string argument is a lookup key (``sys.modules.get("pytest")``, …).
LOOKUP_CALLS = frozenset(
    {
        "get",
        "getenv",
        "pop",
        "setdefault",
        "import_module",
        "find_spec",
        "__import__",
        "__contains__",
        "endswith",
        "startswith",
        "find",
        "index",
        "count",
    }
)
JUSTIFICATION_RE = re.compile(r"^\s*test change\s*:\s*\S", re.IGNORECASE | re.MULTILINE)
UNJUSTIFIED_MESSAGE = (
    "changed test expectations must be justified with a 'Test change:' line explaining why "
    "the old expectation was wrong"
)
MODULE_SCOPE = "<module>"

_FuncDef = ast.FunctionDef | ast.AsyncFunctionDef


# ---- models ---------------------------------------------------------------------------------


class Violation(BaseModel):
    """One broken integrity rule."""

    rule: str
    path: str
    line: int | None = None
    message: str

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.line else self.path
        return f"[{self.rule}] {where}: {self.message}"


class ExpectationChange(BaseModel):
    """A changed expectation in a test (needs a ``Test change:`` justification)."""

    path: str
    test: str
    line: int | None = None
    before: str
    after: str

    def describe(self) -> str:
        """One line: ``path::test: before → after``."""
        return f"{self.path}::{self.test}: {_one_line(self.before)} → {_one_line(self.after)}"


class PatchReport(BaseModel):
    """Result of :func:`check_patch`."""

    violations: list[Violation] = Field(default_factory=list)
    expectation_changes: list[ExpectationChange] = Field(default_factory=list)
    test_files_changed: list[str] = Field(default_factory=list)
    source_files_changed: list[str] = Field(default_factory=list)
    # Test files other tests may depend on: conftest.py, test-dir modules without tests
    # (helpers, factories) and test data. Changing them triggers the regression run.
    shared_test_files_changed: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def rule_ids(self) -> list[str]:
        return sorted({v.rule for v in self.violations})

    def summary(self) -> str:
        """Human-readable list of the violations (the rejection reason for the fixer)."""
        if self.ok:
            return "patch check passed"
        lines = [f"{len(self.violations)} integrity violation(s):"]
        lines += [f"- {v}" for v in self.violations]
        return "\n".join(lines)


# ---- entry point ----------------------------------------------------------------------------


class CheckContext(BaseModel):
    """Extra input for :func:`check_patch`: the failing test's file(s) at HEAD."""

    test_files: dict[str, str] = Field(default_factory=dict)  # path → source


def is_test_file(path: str, layout: TestLayout = DEFAULT_LAYOUT) -> bool:
    """True for ``.py`` files that get test-file rules (see :mod:`.test_layout`)."""
    return path.endswith(".py") and layout.is_test_path(path)


def has_test_change_justification(explanation: str) -> bool:
    return bool(JUSTIFICATION_RE.search(explanation or ""))


def check_patch(
    repo: GitRepo, explanation: str, context: CheckContext | None = None
) -> PatchReport:
    """Check the working-tree changes against ``HEAD`` for integrity violations."""
    report = PatchReport()
    layout = read_layout(repo)
    plugins: set[str] | None = None  # computed lazily: only needed for changed source files
    literals = _test_literals(context)
    changed = repo.changed_files()
    new_modules = _module_names(
        p for p in changed if p.endswith(".py") and repo.file_bytes_at("HEAD", p) is None
    )
    for path in changed:
        name = PurePosixPath(path).name
        if name in CONFIG_FILES:
            before, after = repo.file_at("HEAD", path), _read_worktree(repo, path, report)
            report.violations += check_config_file(path, before, after)
        elif path.endswith(".py"):
            before, after = repo.file_at("HEAD", path), _read_worktree(repo, path, report)
            if layout.is_test_path(path):
                report.test_files_changed.append(path)
                if _is_shared_test_module(path, before, after):
                    report.shared_test_files_changed.append(path)
                violations, changes = check_test_file(path, before, after, new_modules)
                report.violations += violations
                report.expectation_changes += changes
            else:
                report.source_files_changed.append(path)
                if plugins is None:
                    plugins = head_plugin_paths(repo, layout) | _new_plugin_paths(repo, changed)
                report.violations += check_source_file(
                    path, before, after, literals, is_plugin=path in plugins
                )
        elif layout.in_test_dir(path):
            report.test_files_changed.append(path)
            report.shared_test_files_changed.append(path)
            violation, change = check_test_data(
                path, repo.file_bytes_at("HEAD", path), _read_bytes(repo, path)
            )
            report.violations += violation
            report.expectation_changes += change
    if report.expectation_changes and not has_test_change_justification(explanation):
        for path in dict.fromkeys(c.path for c in report.expectation_changes):
            report.violations.append(
                Violation(rule="unjustified_test_change", path=path, message=UNJUSTIFIED_MESSAGE)
            )
    return report


def _is_shared_test_module(path: str, before: str | None, after: str | None) -> bool:
    """conftest.py, or a module in the test tree that defines no tests (helpers, factories)."""
    if PurePosixPath(path).name == "conftest.py":
        return True
    versions = [_TestModule.build(src, _parse_quiet(src, path)) for src in (before, after) if src]
    return all(not (mod.tests or mod.classes) for mod in versions)


def _read_bytes(repo: GitRepo, path: str) -> bytes | None:
    file = repo.path / path
    try:
        return file.read_bytes() if file.is_file() else None
    except OSError:
        return None


def _read_worktree(repo: GitRepo, path: str, report: PatchReport) -> str | None:
    file = repo.path / path
    try:
        if not file.is_file():
            return None
        return file.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        report.violations.append(
            Violation(rule="unreadable", path=path, message=f"cannot read the file: {exc}")
        )
        return None


def _new_plugin_paths(repo: GitRepo, changed: list[str]) -> set[str]:
    """Plugins registered by ``pytest_plugins`` in the changed files' new content."""
    modules: set[str] = set()
    for path in changed:
        if path.endswith(".py"):
            text = _read_worktree(repo, path, PatchReport())
            if text and "pytest_plugins" in text:
                modules |= pytest_plugins_in(text)
    return plugin_paths(modules)


# ---- shared AST helpers ---------------------------------------------------------------------


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _dotted(node: ast.AST) -> str:
    """``a.b.c`` for Name/Attribute chains (a Call uses its func); "" otherwise."""
    if isinstance(node, ast.Call):
        return _dotted(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _last_name(node: ast.AST) -> str:
    return _dotted(node).rsplit(".", 1)[-1]


def _dump(node: ast.AST | None) -> str:
    return ast.dump(node) if node is not None else ""


def _segment(src: str, node: ast.AST) -> str:
    return ast.get_source_segment(src, node) or ast.unparse(node)


def _parse(src: str | None) -> ast.Module | None:
    """Parse ``src`` (None → None). Raises SyntaxError."""
    return ast.parse(src) if src is not None else None


def _parse_quiet(src: str | None, path: str) -> ast.Module | None:
    """Parse the HEAD version; an unparseable HEAD file can't be compared (treated as new)."""
    try:
        return _parse(src)
    except _PARSE_ERRORS:
        log.debug("[integrity] %s does not parse at HEAD; comparing against an empty file", path)
        return None


def _scoped_walk(tree: ast.AST | None) -> Iterator[tuple[str, ast.AST]]:
    """Every node with its enclosing function/class path; decorators belong to their def."""
    if tree is None:
        return

    def rec(node: ast.AST, scope: str) -> Iterator[tuple[str, ast.AST]]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                inner = child.name if scope == MODULE_SCOPE else f"{scope}::{child.name}"
                yield inner, child
                yield from rec(child, inner)
            else:
                yield scope, child
                yield from rec(child, scope)

    yield from rec(tree, MODULE_SCOPE)


def _new_sites(
    old: Sequence[tuple[Hashable, int]], new: Sequence[tuple[Hashable, int]]
) -> list[tuple[Hashable, int]]:
    """Keys that occur more often in ``new`` than in ``old``, with their last line in ``new``."""
    before = Counter(k for k, _ in old)
    after = Counter(k for k, _ in new)
    out = []
    for key, count in after.items():
        if count > before[key]:
            out.append((key, max(line for k, line in new if k == key)))
    return sorted(out, key=lambda kv: kv[1])


# ---- test-file structure --------------------------------------------------------------------


def _is_test_class(node: ast.ClassDef) -> bool:
    return node.name.startswith("Test") or any(
        _last_name(b).endswith("TestCase") for b in node.bases
    )


def _is_fixture(node: _FuncDef) -> bool:
    return any(_last_name(d) == "fixture" for d in node.decorator_list)


@dataclass
class _TestModule:
    """Tests, test classes and fixtures of one version of a test file."""

    src: str = ""
    tree: ast.Module | None = None
    tests: dict[str, _FuncDef] = field(default_factory=dict)
    classes: dict[str, ast.ClassDef] = field(default_factory=dict)
    fixtures: dict[str, _FuncDef] = field(default_factory=dict)
    setups: dict[str, _FuncDef] = field(default_factory=dict)  # setUp, setup_method, …

    @classmethod
    def build(cls, src: str | None, tree: ast.Module | None) -> _TestModule:
        mod = cls(src=src or "", tree=tree)
        if tree is not None:
            mod._collect(tree.body, "", in_test_class=False)
        return mod

    def _collect(self, body: list[ast.stmt], prefix: str, in_test_class: bool) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                key = prefix + node.name
                if _is_fixture(node):
                    self.fixtures[key] = node
                elif node.name in SETUP_NAMES:
                    self.setups[key] = node
                elif node.name.startswith("test") and (not prefix or in_test_class):
                    self.tests[key] = node
            elif isinstance(node, ast.ClassDef) and _is_test_class(node):
                key = prefix + node.name
                self.classes[key] = node
                self._collect(node.body, key + "::", in_test_class=True)


# ---- test-file rules ------------------------------------------------------------------------


def check_test_file(
    path: str,
    before: str | None,
    after: str | None,
    new_modules: frozenset[str] = frozenset(),
) -> tuple[list[Violation], list[ExpectationChange]]:
    """Rules for a test file (``conftest.py`` included); see the module docstring.

    ``new_modules`` are dotted names of ``.py`` files created by this patch (imports of
    code-under-test names from them are ``code_under_test_patched``).
    """
    try:
        new_tree = _parse(after)
    except _PARSE_ERRORS as exc:
        return [_parse_violation(path, exc)], []
    old = _TestModule.build(before, _parse_quiet(before, path))
    new = _TestModule.build(after, new_tree)

    violations = _removed_tests(path, old, new)
    violations += _removed_fixtures(path, old, new)
    violations += _added_skips(path, old, new)
    violations += _removed_params(path, old, new)
    stub_violations, setup_changes = _fixture_changes(path, old, new)
    violations += stub_violations
    violations += _collection_tampering(path, old.tree, new.tree)
    violations += _patching(path, old.tree, new.tree)
    import_violations, changes = _import_changes(path, old, new, new_modules)
    violations += import_violations
    for key, new_fn in new.tests.items():
        old_fn = old.tests.get(key)
        violations += _trivial_assertions(path, key, old_fn, new_fn)
        if old_fn is not None:
            test_violations, test_changes = _compare_test(path, key, old, old_fn, new, new_fn)
            violations += test_violations
            changes += test_changes
    changes += _module_assignment_changes(path, old, new)
    changes += setup_changes
    return violations, changes


def _is_stand_in(expr: ast.expr | None) -> bool:
    """A dummy value: constant/None, empty literal, lambda, ``object()``, ``Mock()``, …"""
    if expr is None or isinstance(expr, (ast.Constant, ast.Lambda)):
        return True
    if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
        return not expr.elts
    if isinstance(expr, ast.Dict):
        return not expr.keys
    if isinstance(expr, ast.Call):
        name = _last_name(expr)
        if name in STAND_IN_CALLS:
            return True
        return name in EMPTY_CALLS and not expr.args and not expr.keywords
    return False


def _stand_in_value(stmt: ast.stmt) -> ast.expr | None | bool:
    """The value a ``return``/``yield``/assignment statement provides, False if it is none."""
    if isinstance(stmt, ast.Return):
        return stmt.value
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, (ast.Yield, ast.YieldFrom)):
        return stmt.value.value
    if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
        return stmt.value
    return False


def _only_stand_ins(fn: _FuncDef) -> bool:
    """The body does nothing but return/yield/assign dummy stand-ins (or nothing at all)."""
    for stmt in fn.body:
        if isinstance(stmt, ast.Pass) or (
            isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
        ):
            continue  # pass, ``...``, docstring
        value = _stand_in_value(stmt)
        if value is False or not _is_stand_in(value):  # type: ignore[arg-type]
            return False
    return True


def _provides_stand_in(fn: _FuncDef) -> bool:
    """Some ``return``/``yield`` in ``fn`` hands out a dummy stand-in."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and _is_stand_in(node.value):
            return True
        if isinstance(node, (ast.Yield, ast.YieldFrom)) and _is_stand_in(node.value):
            return True
    return False


def _is_stubbed(old_fn: _FuncDef, new_fn: _FuncDef) -> bool:
    if _only_stand_ins(new_fn) and not _only_stand_ins(old_fn):
        return True
    raised_before = any(isinstance(n, ast.Raise) for n in ast.walk(old_fn))
    raises_now = any(isinstance(n, ast.Raise) for n in ast.walk(new_fn))
    return raised_before and not raises_now and _provides_stand_in(new_fn)


def _fixture_changes(
    path: str, old: _TestModule, new: _TestModule
) -> tuple[list[Violation], list[ExpectationChange]]:
    """``fixture_stubbed`` for stubbed fixtures/setups; other changes are expectation changes."""
    violations: list[Violation] = []
    changes: list[ExpectationChange] = []
    for old_items, new_items in ((old.fixtures, new.fixtures), (old.setups, new.setups)):
        for key, new_fn in new_items.items():
            old_fn = old_items.get(key)
            if old_fn is None or _dump(old_fn) == _dump(new_fn):
                continue
            if _is_stubbed(old_fn, new_fn):
                violations.append(
                    Violation(
                        rule="fixture_stubbed",
                        path=path,
                        line=new_fn.lineno,
                        message=f"{key}: {STUBBED_MESSAGE}",
                    )
                )
                continue
            changes.append(
                ExpectationChange(
                    path=path,
                    test=key,
                    line=new_fn.lineno,
                    before=_segment(old.src, old_fn),
                    after=_segment(new.src, new_fn),
                )
            )
    return violations, changes


def _parse_violation(path: str, exc: Exception) -> Violation:
    if isinstance(exc, SyntaxError):
        return Violation(
            rule="syntax_error", path=path, line=exc.lineno, message=f"does not parse: {exc.msg}"
        )
    return Violation(
        rule="unparseable",
        path=path,
        message=f"cannot be parsed ({type(exc).__name__}); simplify the change",
    )


def _removed_tests(path: str, old: _TestModule, new: _TestModule) -> list[Violation]:
    out = [
        Violation(
            rule="test_removed",
            path=path,
            line=old.tests[key].lineno,
            message=f"test {key} was deleted or renamed; tests must not be removed",
        )
        for key in old.tests
        if key not in new.tests
    ]
    out += [
        Violation(
            rule="test_removed",
            path=path,
            line=old.classes[key].lineno,
            message=f"test class {key} was deleted or renamed",
        )
        for key in old.classes
        if key not in new.classes
    ]
    return out


def _removed_fixtures(path: str, old: _TestModule, new: _TestModule) -> list[Violation]:
    out = [
        Violation(
            rule="fixture_removed",
            path=path,
            line=old.fixtures[key].lineno,
            message=f"fixture {key} was deleted",
        )
        for key in old.fixtures
        if key not in new.fixtures
    ]
    for key, new_fn in new.tests.items():
        old_fn = old.tests.get(key)
        if old_fn is None:
            continue
        gone = [p for p in _params(old_fn) if p not in _params(new_fn)]
        if gone:
            out.append(
                Violation(
                    rule="fixture_removed",
                    path=path,
                    line=new_fn.lineno,
                    message=f"test {key} no longer takes {', '.join(gone)}",
                )
            )
    return out


def _params(fn: _FuncDef) -> list[str]:
    args = fn.args
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    return [n for n in names if n not in ("self", "cls")]


def _skip_sites(tree: ast.Module | None) -> list[tuple[Hashable, int]]:
    sites: list[tuple[Hashable, int]] = []
    for scope, node in _scoped_walk(tree):
        name = ""
        if isinstance(node, ast.Attribute) and node.attr in SKIP_NAMES:
            name = node.attr
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id if node.func.id in SKIP_NAMES else ""
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in node.decorator_list:  # bare ``@skip``-style decorators
                if isinstance(dec, ast.Name) and dec.id in SKIP_NAMES:
                    sites.append((f"{dec.id} in {scope}", dec.lineno))
        if name:
            sites.append((f"{name} in {scope}", node.lineno))
    return sites


def _added_skips(path: str, old: _TestModule, new: _TestModule) -> list[Violation]:
    return [
        Violation(
            rule="skip_added",
            path=path,
            line=line,
            message=f"new {key}; tests must not be skipped or xfailed",
        )
        for key, line in _new_sites(_skip_sites(old.tree), _skip_sites(new.tree))
    ]


def _parametrize_cases(node: _FuncDef | ast.ClassDef) -> dict[str, int | None]:
    """argnames dump → number of literal argvalues (None if not a literal)."""
    out: dict[str, int | None] = {}
    for dec in node.decorator_list:
        if not (isinstance(dec, ast.Call) and _last_name(dec) == "parametrize"):
            continue
        argnames = dec.args[0] if dec.args else _kwarg(dec, "argnames")
        argvalues = dec.args[1] if len(dec.args) > 1 else _kwarg(dec, "argvalues")
        count = (
            len(argvalues.elts) if isinstance(argvalues, (ast.List, ast.Tuple, ast.Set)) else None
        )
        out[_dump(argnames)] = count
    return out


def _kwarg(call: ast.Call, name: str) -> ast.expr | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


def _removed_params(path: str, old: _TestModule, new: _TestModule) -> list[Violation]:
    out = []
    old_items = {**old.tests, **old.classes}
    new_items = {**new.tests, **new.classes}
    for key, old_node in old_items.items():
        new_node = new_items.get(key)
        if new_node is None:
            continue  # reported as test_removed
        new_cases = _parametrize_cases(new_node)
        for argnames, old_count in _parametrize_cases(old_node).items():
            if old_count is None:
                continue
            new_count = new_cases.get(argnames, 0)
            if new_count is not None and new_count < old_count:
                out.append(
                    Violation(
                        rule="param_removed",
                        path=path,
                        line=new_node.lineno,
                        message=f"{key}: parametrize cases went from {old_count} to {new_count}",
                    )
                )
    return out


def _collect_ignore_stmts(tree: ast.Module | None) -> list[str]:
    out = []
    for node in ast.walk(tree) if tree is not None else ():
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            targets = [node.func.value]
        if any(isinstance(t, ast.Name) and t.id in COLLECT_IGNORE_NAMES for t in targets):
            out.append(_dump(node))
    return out


def _hooks(tree: ast.Module | None) -> dict[str, _FuncDef]:
    if tree is None:
        return {}
    return {
        n.name: n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in TAMPERING_HOOKS
    }


def _collection_tampering(
    path: str, old_tree: ast.Module | None, new_tree: ast.Module | None
) -> list[Violation]:
    out = []
    added = Counter(_collect_ignore_stmts(new_tree)) - Counter(_collect_ignore_stmts(old_tree))
    if added:
        out.append(
            Violation(
                rule="collection_tampering",
                path=path,
                message="collect_ignore/collect_ignore_glob/pytest_plugins changed",
            )
        )
    old_hooks = _hooks(old_tree)
    for name, fn in _hooks(new_tree).items():
        if name not in old_hooks:
            message = f"pytest hook {name} added; hooks can hide test failures"
        elif _dump(old_hooks[name]) != _dump(fn):
            message = f"pytest hook {name} changed; hooks can hide test failures"
        else:
            continue
        out.append(
            Violation(rule="collection_tampering", path=path, line=fn.lineno, message=message)
        )
    return out


_STDLIB = frozenset(sys.stdlib_module_names) | {"pytest", "_pytest", "mock"}


def _bindings(tree: ast.Module | None) -> dict[str, str]:
    """Module-level imported name → where it comes from (``module:name``)."""
    out: dict[str, str] = {}
    for node in tree.body if tree is not None else ():
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".")[0]
                out[bound] = alias.name if alias.asname else bound
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                out[alias.asname or alias.name] = f"{module}:{alias.name}"
    return out


def _root_module(origin: str) -> str:
    return origin.split(":", 1)[0].split(".")[0]


def _is_patch_call(node: ast.Call) -> str:
    """A description if ``node`` patches something, else ""."""
    parts = _dotted(node).split(".")
    if isinstance(node.func, ast.Attribute) and parts[-1] in PATCH_METHODS:
        return f"{parts[-2] if len(parts) > 1 else ''}.{parts[-1]}(...)".lstrip(".")
    if isinstance(node.func, ast.Name) and node.func.id in ("setattr", "delattr"):
        return f"{node.func.id}(...)"
    if parts[-1] == "patch" or (len(parts) > 1 and parts[-2] == "patch"):
        return f"{'.'.join(parts)}(...)"
    return ""


def _rebinding(target: ast.expr, imported: set[str]) -> str:
    """A description if assigning to ``target`` replaces imported code, else ""."""
    if isinstance(target, ast.Name) and target.id in imported:
        return f"rebinding of imported {target.id}"
    if isinstance(target, ast.Attribute):
        root = target
        while isinstance(root, ast.Attribute):
            root = root.value  # type: ignore[assignment]
        if isinstance(root, ast.Name) and root.id in imported:
            return f"assignment to {ast.unparse(target)}"
    return ""


def _patch_sites(tree: ast.Module | None) -> list[tuple[Hashable, int]]:
    imported = set(_bindings(tree))
    sites: list[tuple[Hashable, int]] = []
    for scope, node in _scoped_walk(tree):
        descs: list[str] = []
        if isinstance(node, ast.Call):
            descs.append(_is_patch_call(node))
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            descs += [_rebinding(t, imported) for t in targets]
        sites += [(f"{d} in {scope}", node.lineno) for d in descs if d]  # type: ignore[attr-defined]
    return sites


def _patching(
    path: str, old_tree: ast.Module | None, new_tree: ast.Module | None
) -> list[Violation]:
    """``code_under_test_patched``: new patching or rebinding (imports: ``_import_changes``)."""
    out = [
        Violation(
            rule="code_under_test_patched",
            path=path,
            line=line,
            message=f"new {desc}; tests must exercise the real code, not a substitute",
        )
        for desc, line in _new_sites(_patch_sites(old_tree), _patch_sites(new_tree))
    ]
    return out


def _module_names(paths: Iterable[str]) -> frozenset[str]:
    """Dotted module names of ``paths``, plus every trailing part (``tests.fake`` → ``fake``),
    since tests often import by a shorter name (rootdir/``sys.path`` insertion)."""
    names: set[str] = set()
    for path in paths:
        parts = path.removesuffix(".py").split("/")
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names |= {".".join(parts[i:]) for i in range(len(parts)) if parts[i:]}
    return frozenset(names)


def _is_substitute_origin(origin: str, new_modules: frozenset[str]) -> bool:
    """``origin`` (``module:name``) is a mock module or a module created by this patch."""
    module = origin.split(":", 1)[0].lstrip(".")
    if module in ("mock", "unittest.mock") or module.startswith(("mock.", "unittest.mock.")):
        return True
    return module in new_modules


def _import_segments(mod: _TestModule, names: set[str]) -> str:
    """Source of the module-level imports in ``mod`` that bind any of ``names``."""
    out = []
    for node in mod.tree.body if mod.tree is not None else ():
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            bound = {a.asname or a.name.split(".")[0] for a in node.names}
            if isinstance(node, ast.ImportFrom):
                bound = {a.asname or a.name for a in node.names}
            if bound & names:
                out.append(_segment(mod.src, node))
    return "\n".join(out)


def _import_changes(
    path: str, old: _TestModule, new: _TestModule, new_modules: frozenset[str]
) -> tuple[list[Violation], list[ExpectationChange]]:
    """Changed/removed imports of non-stdlib code are expectation changes (an import fix after
    a rename is legitimate); importing from a mock module or from a file this patch created is
    ``code_under_test_patched``."""
    old_bindings, new_bindings = _bindings(old.tree), _bindings(new.tree)
    violations: list[Violation] = []
    changed: set[str] = set()
    for name, now in new_bindings.items():
        before = old_bindings.get(name)
        if now != before and _is_substitute_origin(now, new_modules):
            violations.append(
                Violation(
                    rule="code_under_test_patched",
                    path=path,
                    message=f"{name} is now imported from {now.split(':', 1)[0]} (a mock or a "
                    "module created by this fix); tests must exercise the real code",
                )
            )
    for name, origin in old_bindings.items():
        now = new_bindings.get(name)
        if now == origin or _root_module(origin) in _STDLIB:
            continue
        if now is None or not _is_substitute_origin(now, new_modules):
            changed.add(name)
    if not changed:
        return violations, []
    added = {
        n
        for n, o in new_bindings.items()
        if n not in old_bindings and _root_module(o) not in _STDLIB
    }
    change = ExpectationChange(
        path=path,
        test="imports",
        before=_import_segments(old, changed),
        after=_import_segments(new, changed | added) or "(removed)",
    )
    return violations, [change]


def _is_expectation_call(node: ast.Call) -> bool:
    name = _last_name(node)
    return name.startswith("assert") or name in EXPECT_CALL_NAMES


def _assertions(fn: _FuncDef) -> list[ast.AST]:
    nodes = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Assert) or (isinstance(n, ast.Call) and _is_expectation_call(n))
    ]
    return sorted(nodes, key=lambda n: (n.lineno, n.col_offset))


def _count(fn: _FuncDef, pred) -> int:  # noqa: ANN001 - small private helper
    return sum(1 for n in ast.walk(fn) if pred(n))


def _is_raises(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and _last_name(node) in RAISES_NAMES


def _is_try(node: ast.AST) -> bool:
    try_types: tuple[type, ...] = (ast.Try, getattr(ast, "TryStar", ast.Try))
    return isinstance(node, try_types) or (
        isinstance(node, ast.Call) and _last_name(node) == "suppress"
    )


def _is_empty_body(body: list[ast.stmt]) -> bool:
    return all(
        isinstance(s, ast.Pass) or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
        for s in body
    )


def _compare_test(
    path: str,
    key: str,
    old: _TestModule,
    old_fn: _FuncDef,
    new: _TestModule,
    new_fn: _FuncDef,
) -> tuple[list[Violation], list[ExpectationChange]]:
    """Rules 4, 6, 7, 8 and expectation changes for a test present before and after."""

    def violation(rule: str, message: str) -> Violation:
        return Violation(rule=rule, path=path, line=new_fn.lineno, message=f"{key}: {message}")

    out: list[Violation] = []
    old_asserts, new_asserts = _assertions(old_fn), _assertions(new_fn)
    if len(new_asserts) < len(old_asserts):
        out.append(
            violation(
                "assertion_removed",
                f"assertions went from {len(old_asserts)} to {len(new_asserts)}",
            )
        )
    if _is_empty_body(new_fn.body) and not _is_empty_body(old_fn.body):
        out.append(violation("test_emptied", "test body was emptied"))
    if _count(new_fn, _is_raises) and not _count(old_fn, _is_raises):
        out.append(
            violation(
                "expects_exception_added",
                "now expects an exception (pytest.raises/assertRaises); fix the error instead",
            )
        )
    if _count(new_fn, _is_try) and not _count(old_fn, _is_try):
        out.append(violation("try_added", "try/except (or suppress) added to the test"))
    if len(new_asserts) < len(old_asserts):
        return out, []
    changes = _paired_changes(old.src, old_asserts, new.src, new_asserts)
    changes += _paired_changes(old.src, _tolerances(old_fn), new.src, _tolerances(new_fn))
    changes += _paired_changes(
        old.src,
        _expected_assignments(old_fn),
        new.src,
        _expected_assignments(new_fn),
        key=_literals,
    )
    return out, [
        ExpectationChange(path=path, test=key, line=line, before=b, after=a)
        for b, a, line in changes
    ]


def _literals(node: ast.AST) -> str:
    """The literal values in ``node`` (renaming variables keeps this the same)."""
    return repr([n.value for n in ast.walk(node) if isinstance(n, ast.Constant)])


def _paired_changes(
    old_src: str,
    old_nodes: list[ast.AST],
    new_src: str,
    new_nodes: list[ast.AST],
    key=_dump,  # noqa: ANN001 - what counts as "changed" (whole AST or only its literals)
) -> list[tuple[str, str, int | None]]:
    """(before, after, line) for nodes that changed, paired by a diff of ``key(node)``."""
    old_dumps = [key(n) for n in old_nodes]
    new_dumps = [key(n) for n in new_nodes]
    out: list[tuple[str, str, int | None]] = []
    matcher = difflib.SequenceMatcher(a=old_dumps, b=new_dumps, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag not in ("replace", "delete"):
            continue  # "insert" = an added expectation: not a changed one
        olds, news = old_nodes[i1:i2], new_nodes[j1:j2]
        for k, old_node in enumerate(olds):
            if k < len(news):
                new_node = news[k]
                out.append(
                    (_segment(old_src, old_node), _segment(new_src, new_node), new_node.lineno)
                )
            else:
                out.append((_segment(old_src, old_node), "(removed)", None))
    return out


def _tolerances(fn: _FuncDef) -> list[ast.AST]:
    """Calls with tolerance keywords that are not inside an assertion (those are compared)."""
    inside: set[int] = set()
    for a in _assertions(fn):
        inside.update(id(n) for n in ast.walk(a) if n is not a)
    nodes = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and id(n) not in inside
        and not _is_expectation_call(n)
        and any(k.arg in TOLERANCE_KWARGS for k in n.keywords)
    ]
    return sorted(nodes, key=lambda n: (n.lineno, n.col_offset))


def _expected_assignments(fn: _FuncDef) -> list[ast.AST]:
    """Assignments to names that the test's assertions read (e.g. ``expected = 5``)."""
    used = {n.id for a in _assertions(fn) for n in ast.walk(a) if isinstance(n, ast.Name)}
    nodes: list[ast.AST] = []
    for n in ast.walk(fn):
        targets: list[ast.expr] = []
        if isinstance(n, ast.Assign):
            targets = n.targets
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)):
            targets = [n.target]
        names = {t.id for target in targets for t in ast.walk(target) if isinstance(t, ast.Name)}
        if names & used:
            nodes.append(n)
    return sorted(nodes, key=lambda n: (n.lineno, n.col_offset))


def _module_assignments(tree: ast.Module | None) -> dict[str, ast.stmt]:
    out: dict[str, ast.stmt] = {}
    for node in tree.body if tree is not None else ():
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name) and t.id not in ("pytestmark", *COLLECT_IGNORE_NAMES):
                    out[t.id] = node
    return out


def _module_assignment_changes(
    path: str, old: _TestModule, new: _TestModule
) -> list[ExpectationChange]:
    """Changed module-level constants in a test file (e.g. ``EXPECTED = ...``)."""
    old_assigns = _module_assignments(old.tree)
    out = []
    for name, node in _module_assignments(new.tree).items():
        prev = old_assigns.get(name)
        if prev is not None and _literals(prev.value) != _literals(node.value):  # type: ignore[attr-defined]
            out.append(
                ExpectationChange(
                    path=path,
                    test=MODULE_SCOPE,
                    line=node.lineno,
                    before=_segment(old.src, prev),
                    after=_segment(new.src, node),
                )
            )
    return out


def _always_true(expr: ast.expr) -> bool:
    if isinstance(expr, ast.Constant):
        return bool(expr.value)
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return isinstance(expr.operand, ast.Constant) and not expr.operand.value
    if isinstance(expr, ast.BoolOp) and isinstance(expr.op, ast.Or):
        return any(_always_true(v) for v in expr.values)
    if isinstance(expr, ast.Compare) and len(expr.ops) == 1:
        if isinstance(expr.ops[0], (ast.Eq, ast.Is, ast.GtE, ast.LtE)):
            return _dump(expr.left) == _dump(expr.comparators[0])
    return False


def _is_trivial(node: ast.AST) -> bool:
    if isinstance(node, ast.Assert):
        return _always_true(node.test)
    if not isinstance(node, ast.Call) or not node.args:
        return False
    name, first = _last_name(node), node.args[0]
    if name == "assertTrue":
        return _always_true(first)
    if name == "assertFalse":
        return isinstance(first, ast.Constant) and not first.value
    if name == "assertIsNotNone":
        return isinstance(first, ast.Constant) and first.value is not None
    if name in ("assertEqual", "assertIs") and len(node.args) >= 2:
        return _dump(first) == _dump(node.args[1])
    return False


def _trivial_assertions(
    path: str, key: str, old_fn: _FuncDef | None, new_fn: _FuncDef
) -> list[Violation]:
    before = sum(1 for n in _assertions(old_fn) if _is_trivial(n)) if old_fn else 0
    trivial = [n for n in _assertions(new_fn) if _is_trivial(n)]
    if len(trivial) <= before:
        return []
    return [
        Violation(
            rule="trivial_assertion",
            path=path,
            line=trivial[-1].lineno,
            message=f"{key}: always-true assertion added",
        )
    ]


# ---- source-file rules ----------------------------------------------------------------------


def check_source_file(
    path: str,
    before: str | None,
    after: str | None,
    test_literals: frozenset[tuple[str, object]] = frozenset(),
    is_plugin: bool = False,
) -> list[Violation]:
    """Rules for a non-test ``.py`` file; see the module docstring.

    ``test_literals`` are the failing test's literals (``special_case_inputs``);
    ``is_plugin`` adds the hook rules for modules registered via ``pytest_plugins``.
    """
    try:
        new_tree = _parse(after)
    except _PARSE_ERRORS as exc:
        return [_parse_violation(path, exc)]
    old_tree = _parse_quiet(before, path)
    out = _special_cases(path, old_tree, new_tree, test_literals)
    out += _hardcoded_returns(path, old_tree, new_tree)
    if is_plugin:
        out += _collection_tampering(path, old_tree, new_tree)
    out += _source_detection(path, old_tree, new_tree)
    return out


def _source_detection(
    path: str, old_tree: ast.Module | None, new_tree: ast.Module | None
) -> list[Violation]:
    """``test_detection`` and ``error_swallowed``."""
    out = [
        Violation(rule="test_detection", path=path, line=line, message=f"new {desc}")
        for desc, line in _new_sites(_detection_sites(old_tree), _detection_sites(new_tree))
    ]
    out += [
        Violation(
            rule="error_swallowed",
            path=path,
            line=line,
            message=f"new broad except that swallows the error in {scope}",
        )
        for scope, line in _new_sites(_swallow_sites(old_tree), _swallow_sites(new_tree))
    ]
    return out


def _env_key(node: ast.AST) -> ast.expr | None:
    """The key expression of an ``os.environ``/``getenv`` lookup, if ``node`` is one."""
    if isinstance(node, ast.Subscript) and _dotted(node.value).endswith("environ"):
        return node.slice
    if isinstance(node, ast.Call) and node.args:
        name = _dotted(node)
        if name.endswith(("environ.get", "getenv", "environ.setdefault", "environ.pop")):
            return node.args[0]
    if isinstance(node, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in node.ops):
        if any(_dotted(c).endswith("environ") for c in node.comparators):
            return node.left
    return None


def _detection_sites(tree: ast.Module | None) -> list[tuple[Hashable, int]]:
    """(scope, description) of every construct that could detect a test run."""
    lookups = _lookup_constants(tree)
    sites: list[tuple[Hashable, int]] = []
    for scope, node in _scoped_walk(tree):
        desc = ""
        if isinstance(node, ast.Import):
            if any(a.name.split(".")[0] in ("pytest", "_pytest") for a in node.names):
                desc = "import of pytest"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in ("pytest", "_pytest"):
                desc = "import from pytest"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            low = node.value.lower()
            if id(node) in lookups and ("pytest" in low or "ci_fix" in low):
                desc = f"reference to {node.value!r} (test detection)"
        elif isinstance(node, ast.Name) and "PYTEST" in node.id.upper():
            desc = f"reference to {node.id} (test detection)"
        elif isinstance(node, ast.Attribute) and "PYTEST" in node.attr.upper():
            desc = f"reference to {node.attr} (test detection)"
        else:
            key = _env_key(node)
            if (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and key.value.upper() == "TESTING"
            ):
                desc = f"environment lookup of {key.value!r} (test detection)"
        if desc:
            sites.append((f"{desc} in {scope}", getattr(node, "lineno", 0)))
    return sites


def _lookup_constants(tree: ast.Module | None) -> set[int]:
    """ids of constants used as a comparison operand, subscript key or lookup-call argument."""
    ids: set[int] = set()
    for node in ast.walk(tree) if tree is not None else ():
        operands: list[ast.AST] = []
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            for operand in list(operands):  # ``x in ("pytest", ...)``
                if isinstance(operand, (ast.Tuple, ast.List, ast.Set)):
                    operands += operand.elts
        elif isinstance(node, ast.Subscript):
            operands = [node.slice]
        elif isinstance(node, ast.Call) and _last_name(node) in LOOKUP_CALLS:
            operands = list(node.args)
        ids |= {id(o) for o in operands if isinstance(o, ast.Constant)}
    return ids


def _literal(node: ast.AST) -> tuple[bool, object]:
    """(True, value) for a constant or a negative number; (False, None) otherwise."""
    if isinstance(node, ast.Constant):
        return True, node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float, complex))
        and not isinstance(node.operand.value, bool)
    ):
        return True, -node.operand.value
    return False, None


def _literal_key(value: object) -> tuple[str, object]:
    return (type(value).__name__, value)


def _literal_values(node: ast.AST) -> Iterator[object]:
    """Every literal value in ``node`` (a negative number counts once, as negative)."""
    ok, value = _literal(node)
    if ok:
        yield value
        return
    for child in ast.iter_child_nodes(node):
        yield from _literal_values(child)


def _test_literals(context: CheckContext | None) -> frozenset[tuple[str, object]]:
    """Literals the failing test file passes as arguments or expects in assertions."""
    keys: set[tuple[str, object]] = set()
    for source in context.test_files.values() if context else ():
        try:
            tree = ast.parse(source)
        except _PARSE_ERRORS:
            continue
        for node in ast.walk(tree):
            parts: list[ast.AST] = []
            if isinstance(node, ast.Call):
                parts = [*node.args, *(k.value for k in node.keywords)]
            elif isinstance(node, ast.Assert):
                parts = [node.test]
            for part in parts:
                for value in _literal_values(part):
                    try:
                        keys.add(_literal_key(value))
                    except TypeError:  # unhashable (never for constants, but be safe)
                        continue
    return frozenset(keys - SENTINEL_LITERALS)


_SPECIAL_OPS = (ast.Eq, ast.NotEq, ast.In, ast.NotIn, ast.Is, ast.IsNot)
_NAME_LIKE = (ast.Name, ast.Attribute, ast.Subscript)


def _compare_literals(node: ast.Compare) -> list[object]:
    """Literal values a name is compared with in ``node`` ([] if it isn't that shape)."""
    if not all(isinstance(op, _SPECIAL_OPS) for op in node.ops):
        return []
    operands = [node.left, *node.comparators]
    if not any(isinstance(o, _NAME_LIKE) for o in operands):
        return []
    values: list[object] = []
    for operand in operands:
        items = operand.elts if isinstance(operand, (ast.Tuple, ast.List, ast.Set)) else [operand]
        for item in items:
            ok, value = _literal(item)
            if ok:
                values.append(value)
    return values


def _special_case_sites(
    tree: ast.Module | None, literals: frozenset[tuple[str, object]]
) -> list[tuple[Hashable, int]]:
    sites: list[tuple[Hashable, int]] = []
    for scope, node in _scoped_walk(tree):
        if isinstance(node, ast.Compare):
            values = _compare_literals(node)
            if any(_literal_key(v) in literals for v in values if _hashable(v)):
                sites.append((f"`{ast.unparse(node)}` in {scope}", node.lineno))
    return sites


def _hashable(value: object) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _special_cases(
    path: str,
    old_tree: ast.Module | None,
    new_tree: ast.Module | None,
    literals: frozenset[tuple[str, object]],
) -> list[Violation]:
    if not literals:
        return []
    new_sites = _new_sites(
        _special_case_sites(old_tree, literals), _special_case_sites(new_tree, literals)
    )
    return [
        Violation(
            rule="special_case_inputs",
            path=path,
            line=line,
            message=f"new comparison {key} uses a value from the failing test; fix the "
            "general logic instead of special-casing test inputs",
        )
        for key, line in new_sites
    ]


def _is_literal_value(node: ast.AST | None) -> bool:
    if node is None:
        return False
    if _literal(node)[0]:
        return True
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_literal_value(e) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(_is_literal_value(k) for k in node.keys) and all(
            _is_literal_value(v) for v in node.values
        )
    return False


def _returns_only_literal(fn: _FuncDef) -> bool:
    body = [
        s
        for s in fn.body
        if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))  # docstring
        and not isinstance(s, ast.Pass)
    ]
    return len(body) == 1 and isinstance(body[0], ast.Return) and _is_literal_value(body[0].value)


def _functions(tree: ast.Module | None) -> dict[str, _FuncDef]:
    return {
        scope: node
        for scope, node in _scoped_walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _hardcoded_returns(
    path: str, old_tree: ast.Module | None, new_tree: ast.Module | None
) -> list[Violation]:
    old_fns = _functions(old_tree)
    out = []
    for key, fn in _functions(new_tree).items():
        old_fn = old_fns.get(key)
        if old_fn is None or _dump(old_fn) == _dump(fn):
            continue
        if _returns_only_literal(fn) and not _returns_only_literal(old_fn):
            out.append(
                Violation(
                    rule="hardcoded_return",
                    path=path,
                    line=fn.lineno,
                    message=f"{key} now only returns a hard-coded value; compute the result",
                )
            )
    return out


def _is_broad(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    types: Iterable[ast.expr] = (
        handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    )
    return any(_last_name(t) in ("Exception", "BaseException") for t in types)


def _swallows(body: list[ast.stmt]) -> bool:
    for stmt in body:
        if isinstance(stmt, (ast.Pass, ast.Continue)):
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue  # ``...`` or a string
        if isinstance(stmt, ast.Return) and (
            stmt.value is None or isinstance(stmt.value, ast.Constant)
        ):
            continue
        return False
    return True


def _swallow_sites(tree: ast.Module | None) -> list[tuple[Hashable, int]]:
    return [
        (scope, node.lineno)
        for scope, node in _scoped_walk(tree)
        if isinstance(node, ast.ExceptHandler) and _is_broad(node) and _swallows(node.body)
    ]


# ---- test data ------------------------------------------------------------------------------


def _text_or_none(data: bytes | None) -> str | None:
    """Decoded text, or None if ``data`` looks binary."""
    if data is None or b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def check_test_data(
    path: str, before: bytes | None, after: bytes | None
) -> tuple[list[Violation], list[ExpectationChange]]:
    """A non-``.py`` file in a test directory: text → expectation change, binary → violation."""
    if before == after:
        return [], []
    old_text, new_text = _text_or_none(before), _text_or_none(after)
    if (before is not None and old_text is None) or (after is not None and new_text is None):
        return [
            Violation(
                rule="test_data_changed",
                path=path,
                message="binary test data changed; test data must not be rewritten",
            )
        ], []
    change = ExpectationChange(
        path=path,
        test=path,
        before=old_text[:DATA_PREVIEW_CHARS] if old_text is not None else "(new file)",
        after=new_text[:DATA_PREVIEW_CHARS] if new_text is not None else "(deleted)",
    )
    return [], [change]


# ---- config-file rule -----------------------------------------------------------------------


def check_config_file(path: str, before: str | None, after: str | None) -> list[Violation]:
    """``test_config_changed`` when the pytest configuration in ``path`` changed."""
    name = PurePosixPath(path).name
    if name in PYTEST_INIS:
        changed = before != after  # any pytest.ini change (even creating one) matters
    else:
        reader = _toml_pytest if name.endswith(".toml") else _ini_pytest
        changed = reader(before) != reader(after)
    if not changed:
        return []
    return [
        Violation(
            rule="test_config_changed",
            path=path,
            message="pytest configuration changed; fixes must not change how tests run",
        )
    ]


def _toml_pytest(text: str | None) -> object:
    if text is None:
        return None
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return ("<unparseable>", text)
    tool = data.get("tool")
    return tool.get("pytest") if isinstance(tool, dict) else None


def _ini_pytest(text: str | None) -> object:
    if text is None:
        return None
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        parser.read_string(text)
    except configparser.Error:
        return ("<unparseable>", text)
    return {s: dict(parser[s]) for s in INI_PYTEST_SECTIONS if parser.has_section(s)}
