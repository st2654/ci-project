"""Triage for parallel fixing: which failing tests are independent of each other.

Pure functions (no git, no pytest). Two failing tests are *related* when their failure
tracebacks mention a common repo file, or when they live in the same test file; related
tests are fixed by the same (sequential) worker, unrelated groups may be fixed in parallel.

``conftest.py`` frames are ignored when linking tests: shared fixtures show up in many
unrelated tracebacks and would glue everything into one group. Grouping that is too fine is
cheap: a candidate patch that conflicts with an accepted one is just retried sequentially.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

# ``path/to/file.py:123: in func`` / ``tests/test_x.py:10: AssertionError`` (pytest long repr)
_PYTEST_LOCATION_RE = re.compile(r"^\s*(?P<path>[^\s:\"'<>]+?\.py):\d+:", re.MULTILINE)
# ``File "/abs/path.py", line 12, in func`` (native Python tracebacks)
_NATIVE_LOCATION_RE = re.compile(r"File \"(?P<path>[^\"]+?\.py)\", line \d+")
# Path components that mean "installed code", never the repo's own files.
_EXCLUDED_PARTS = frozenset(
    {"site-packages", "dist-packages", ".venv", "venv", ".tox", ".nox", "__pypackages__"}
)


def _roots(repo_root: Path) -> list[str]:
    """The repo root as given and resolved (tracebacks may use either spelling)."""
    roots = [os.path.normpath(os.path.abspath(repo_root))]
    try:
        resolved = os.path.normpath(str(Path(repo_root).resolve()))
    except OSError:
        resolved = roots[0]
    if resolved not in roots:
        roots.append(resolved)
    return roots


def _repo_relative(raw: str, roots: Sequence[str]) -> str | None:
    """``raw`` as a repo-relative posix path, or None when it is outside the repo."""
    raw = raw.strip()
    if not raw:
        return None
    if os.path.isabs(raw):
        candidate = os.path.normpath(raw)
        for root in roots:
            if candidate.startswith(root + os.sep):
                rel = os.path.relpath(candidate, root)
                break
        else:
            return None
    else:
        rel = os.path.normpath(raw)
    if rel.startswith("..") or os.path.isabs(rel) or rel == ".":
        return None
    parts = Path(rel).parts
    if any(part in _EXCLUDED_PARTS for part in parts):
        return None
    return Path(rel).as_posix()


def traceback_files(details: str, repo_root: Path) -> set[str]:
    """Repo-relative paths of the ``.py`` files a pytest failure traceback mentions.

    Recognises pytest's ``path.py:123:`` lines and native ``File "path.py", line N`` frames.
    Files outside ``repo_root`` (stdlib, site-packages, virtualenvs) are left out.
    """
    roots = _roots(repo_root)
    found: set[str] = set()
    for regex in (_PYTEST_LOCATION_RE, _NATIVE_LOCATION_RE):
        for match in regex.finditer(details or ""):
            rel = _repo_relative(match.group("path"), roots)
            if rel is not None:
                found.add(rel)
    return found


def _test_file(node_id: str) -> str:
    return node_id.split("::", 1)[0]


def group_failures(
    pending: Sequence[str], details: Mapping[str, str], repo_root: Path
) -> list[list[str]]:
    """Split ``pending`` test ids into groups that can be fixed independently.

    Tests are in the same group when their traceback file sets overlap (transitively) or
    they are in the same test file; ``conftest.py`` frames are ignored for linking. Groups
    are ordered by their first member's position in ``pending``; members keep ``pending``
    order. Tests with no detected files (and no
    test-file sibling) form groups of their own.
    """
    ids = list(dict.fromkeys(pending))
    parent = list(range(len(ids)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    owner: dict[str, int] = {}  # traceback file or "test file" key -> first test index
    for i, nid in enumerate(ids):
        files = traceback_files(details.get(nid, ""), repo_root)
        keys = {f"file:{f}" for f in files if Path(f).name != "conftest.py"}
        keys.add(f"testfile:{_test_file(nid)}")
        for key in keys:
            if key in owner:
                union(owner[key], i)
            else:
                owner[key] = i

    groups: dict[int, list[str]] = {}
    for i, nid in enumerate(ids):
        groups.setdefault(find(i), []).append(nid)
    return sorted(groups.values(), key=lambda g: ids.index(g[0]))
