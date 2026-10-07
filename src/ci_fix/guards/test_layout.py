"""Which files of the target repo are tests: pytest config (at HEAD) + directory conventions.

A ``.py`` file is a *test file* when it is a test module (matches ``python_files``, default
``test_*.py``/``*_test.py``, or is ``conftest.py``) or lives in a test directory: a
directory named ``tests``/``test``/``testing`` anywhere in its path, or under a configured
``testpaths`` entry. Test files only get test-file rules, never source rules.
"""

from __future__ import annotations

import ast
import configparser
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import PurePosixPath

from ci_fix.logging_setup import get_logger
from ci_fix.tools.git import GitRepo

log = get_logger(__name__)

TEST_DIR_NAMES = frozenset({"tests", "test", "testing"})
DEFAULT_PYTHON_FILES = ("test_*.py", "*_test.py")
# Patterns that match every module: only meaningful inside test directories.
_CATCH_ALL_PATTERNS = frozenset({"*.py", "*"})
_GLOB_CHARS = frozenset("*?[")
# pytest's rootdir config files, in the order pytest picks them.
CONFIG_ORDER = ("pytest.ini", ".pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg")
_INI_SECTION = {"pytest.ini": "pytest", ".pytest.ini": "pytest", "tox.ini": "pytest"}


def _norm(entry: str) -> str:
    entry = entry.strip().removeprefix("./").rstrip("/")
    return "" if entry == "." else entry


@dataclass(frozen=True)
class TestLayout:
    """Test-file classification for one repository."""

    __test__ = False

    testpaths: tuple[str, ...] = ()
    python_files: tuple[str, ...] = DEFAULT_PYTHON_FILES

    def in_test_dir(self, path: str) -> bool:
        if any(part in TEST_DIR_NAMES for part in PurePosixPath(path).parts[:-1]):
            return True
        for entry in filter(None, (_norm(t) for t in self.testpaths)):
            if _GLOB_CHARS & set(entry):
                if fnmatch(path, entry) or fnmatch(path, f"{entry}/*"):
                    return True
            elif path == entry or path.startswith(f"{entry}/"):
                return True
        return False

    def is_test_module(self, path: str) -> bool:
        name = PurePosixPath(path).name
        if name == "conftest.py":
            return True
        for pattern in self.python_files:
            if pattern in _CATCH_ALL_PATTERNS:
                continue
            if fnmatch(path if "/" in pattern else name, pattern):
                return True
        return False

    def is_test_path(self, path: str) -> bool:
        return self.is_test_module(path) or self.in_test_dir(path)


DEFAULT_LAYOUT = TestLayout()


def _as_list(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(value.split())
    if isinstance(value, list):
        return tuple(str(v) for v in value)
    return ()


def pytest_options(name: str, text: str) -> dict[str, object] | None:
    """pytest options in config file ``name`` (None = this file holds no pytest config)."""
    if name.endswith(".toml"):
        try:
            tool = tomllib.loads(text).get("tool", {})
        except tomllib.TOMLDecodeError:
            return None
        pytest_table = tool.get("pytest") if isinstance(tool, dict) else None
        if not isinstance(pytest_table, dict):
            return None
        ini = pytest_table.get("ini_options")
        return dict(ini) if isinstance(ini, dict) else dict(pytest_table)
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        parser.read_string(text)
    except configparser.Error:
        return {} if name in ("pytest.ini", ".pytest.ini") else None
    section = _INI_SECTION.get(name, "tool:pytest")
    if parser.has_section(section):
        return dict(parser[section])
    return {} if name in ("pytest.ini", ".pytest.ini") else None  # pytest.ini always wins


def read_layout(repo: GitRepo) -> TestLayout:
    """The layout from the first pytest config file at HEAD (like pytest's rootdir lookup)."""
    for name in CONFIG_ORDER:
        text = repo.file_at("HEAD", name)
        if text is None:
            continue
        options = pytest_options(name, text)
        if options is None:
            continue
        python_files = _as_list(options.get("python_files")) or DEFAULT_PYTHON_FILES
        layout = TestLayout(_as_list(options.get("testpaths")), python_files)
        log.debug("[integrity] test layout from %s: %s", name, layout)
        return layout
    return DEFAULT_LAYOUT


def pytest_plugins_in(source: str) -> set[str]:
    """Module names listed in a ``pytest_plugins = ...`` assignment in ``source``."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return set()
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "pytest_plugins" for t in targets):
                names |= {
                    n.value
                    for n in ast.walk(node.value)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                }
    return names


def plugin_paths(modules: Iterable[str]) -> set[str]:
    """Repo paths a plugin module name may live at (also under ``src/``)."""
    paths: set[str] = set()
    for module in modules:
        base = module.replace(".", "/")
        for prefix in ("", "src/"):
            paths |= {f"{prefix}{base}.py", f"{prefix}{base}/__init__.py"}
    return paths


def head_plugin_paths(repo: GitRepo, layout: TestLayout) -> set[str]:
    """Paths of modules registered as plugins via ``pytest_plugins`` in test files at HEAD."""
    modules: set[str] = set()
    for path in repo.files_containing("pytest_plugins"):
        if layout.is_test_path(path):
            modules |= pytest_plugins_in(repo.file_at("HEAD", path) or "")
    return plugin_paths(modules)
