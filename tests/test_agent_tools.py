"""Tests for ci_fix.agent.tools.RepoTools (slice 4): file tools the Claude fixer can call.

Every tool returns a string and never raises; paths outside the repo or under ``.git/``
are refused.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from conftest import git
from fixtures import copy_sample_repo

from ci_fix.agent.tools import RepoTools

OPS_PY = "src/calc/ops.py"
READ_MAX = 50
NUMBERED_RE = re.compile(r"^\s*(\d+)\| ")
MORE_RE = re.compile(r"more|truncated|remain", re.IGNORECASE)


def is_error(out: str) -> bool:
    return "error" in out.lower() or "refus" in out.lower() or "not allowed" in out.lower()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = copy_sample_repo(tmp_path / "repo")
    git("init", "-q", "-b", "main", cwd=path)
    git("config", "user.email", "tester@example.com", cwd=path)
    git("config", "user.name", "Tester", cwd=path)
    git("config", "commit.gpgsign", "false", cwd=path)
    git("add", "-A", cwd=path)
    git("commit", "-q", "-m", "initial", cwd=path)
    return path


@pytest.fixture
def tools(repo: Path) -> RepoTools:
    return RepoTools(repo, read_max_lines=READ_MAX)


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    """A file outside the repo whose content must never leak or change."""
    path = tmp_path / "outside.txt"
    path.write_text("TOP-SECRET-OUTSIDE\n", encoding="utf-8")
    return path


def numbered(out: str) -> list[int]:
    return [int(m.group(1)) for line in out.splitlines() if (m := NUMBERED_RE.match(line))]


def touched(tools: RepoTools) -> set[str]:
    return {Path(p).as_posix() for p in tools.files_touched}


def write_lines(repo: Path, relpath: str, n: int) -> None:
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"line {i}\n" for i in range(1, n + 1)), encoding="utf-8")


# --------------------------------------------------------------------------- #
# list_files
# --------------------------------------------------------------------------- #


def test_list_files_root_lists_repo_files(tools: RepoTools) -> None:
    out = tools.list_files()
    assert "ops.py" in out
    assert "test_ops.py" in out
    assert "README.md" in out


def test_list_files_never_shows_git_dir(tools: RepoTools) -> None:
    out = tools.list_files()
    assert not any(
        part == ".git" for line in out.splitlines() for part in re.split(r"[\\/\s]", line.strip())
    )
    assert "HEAD" not in out.split()


def test_list_files_subdir(tools: RepoTools) -> None:
    out = tools.list_files("src")
    assert "ops.py" in out
    assert "test_ops.py" not in out
    assert "README.md" not in out


def test_list_files_pattern(tools: RepoTools) -> None:
    out = tools.list_files(".", pattern="*.py")
    assert "ops.py" in out
    assert "README.md" not in out
    assert "pyproject.toml" not in out


def test_list_files_missing_dir_is_error(tools: RepoTools) -> None:
    assert is_error(tools.list_files("no/such/dir"))


@pytest.mark.parametrize("path", ["..", "../..", "/"])
def test_list_files_outside_repo_is_error(tools: RepoTools, path: str) -> None:
    out = tools.list_files(path)
    assert is_error(out)
    assert "outside.txt" not in out


def test_list_files_git_dir_is_error(tools: RepoTools) -> None:
    assert is_error(tools.list_files(".git"))


# --------------------------------------------------------------------------- #
# read_file
# --------------------------------------------------------------------------- #


def test_read_file_numbers_lines(tools: RepoTools, repo: Path) -> None:
    out = tools.read_file(OPS_PY)
    total = len((repo / OPS_PY).read_text(encoding="utf-8").splitlines())
    assert numbered(out) == list(range(1, total + 1))
    assert re.search(r"^\s*1\| \"\"\"Basic arithmetic", out, re.MULTILINE)
    assert "SEEDED BUG: should be a - b" in out
    assert not MORE_RE.search(out.replace("SEEDED BUG", ""))  # whole file fits


def test_read_file_line_format(tools: RepoTools, repo: Path) -> None:
    write_lines(repo, "many.txt", 20)
    out = tools.read_file("many.txt")
    assert re.search(r"^ +12\| line 12$", out, re.MULTILINE)


def test_read_file_range(tools: RepoTools, repo: Path) -> None:
    write_lines(repo, "many.txt", 120)
    out = tools.read_file("many.txt", start_line=10, end_line=15)
    assert numbered(out) == list(range(10, 16))
    assert "line 9\n" not in out
    assert "line 16" not in out


def test_read_file_caps_at_read_max_lines_with_note(tools: RepoTools, repo: Path) -> None:
    write_lines(repo, "many.txt", 120)
    out = tools.read_file("many.txt")
    assert numbered(out) == list(range(1, READ_MAX + 1))
    assert "line 51" not in out
    assert MORE_RE.search(out), "must say that more lines remain"


def test_read_file_cap_applies_to_explicit_range(tools: RepoTools, repo: Path) -> None:
    write_lines(repo, "many.txt", 300)
    out = tools.read_file("many.txt", start_line=1, end_line=300)
    assert len(numbered(out)) <= READ_MAX
    assert MORE_RE.search(out)


def test_read_file_paging_to_the_end(tools: RepoTools, repo: Path) -> None:
    write_lines(repo, "many.txt", 120)
    out = tools.read_file("many.txt", start_line=101)
    assert numbered(out) == list(range(101, 121))
    assert not MORE_RE.search(out)


def test_read_file_start_past_end_is_not_a_crash(tools: RepoTools, repo: Path) -> None:
    write_lines(repo, "many.txt", 5)
    out = tools.read_file("many.txt", start_line=50)
    assert isinstance(out, str)
    assert numbered(out) == []


def test_read_file_missing_is_error(tools: RepoTools) -> None:
    assert is_error(tools.read_file("src/calc/nope.py"))


def test_read_file_directory_is_error(tools: RepoTools) -> None:
    assert is_error(tools.read_file("src"))


def test_read_file_binary_is_error(tools: RepoTools, repo: Path) -> None:
    (repo / "blob.bin").write_bytes(b"\x00\x01\x02binary\x00data" * 10)
    out = tools.read_file("blob.bin")
    assert is_error(out)
    assert "binary" in out.lower()


def test_read_file_too_large_is_error(tools: RepoTools, repo: Path) -> None:
    (repo / "big.txt").write_text(("y" * 99 + "\n") * 11_000, encoding="utf-8")  # ~1.1 MB
    out = tools.read_file("big.txt")
    assert is_error(out)
    assert len(out) < 2000


def test_read_file_does_not_touch(tools: RepoTools) -> None:
    tools.read_file(OPS_PY)
    tools.list_files()
    tools.search_code("def")
    assert touched(tools) == set()


# --------------------------------------------------------------------------- #
# path escapes
# --------------------------------------------------------------------------- #


def test_read_dotdot_escape_is_refused(tools: RepoTools, outside: Path) -> None:
    out = tools.read_file("../outside.txt")
    assert is_error(out)
    assert "TOP-SECRET-OUTSIDE" not in out


def test_read_nested_dotdot_escape_is_refused(tools: RepoTools, outside: Path) -> None:
    out = tools.read_file("src/../../outside.txt")
    assert is_error(out)
    assert "TOP-SECRET-OUTSIDE" not in out


def test_read_absolute_outside_path_is_refused(tools: RepoTools, outside: Path) -> None:
    out = tools.read_file(str(outside))
    assert is_error(out)
    assert "TOP-SECRET-OUTSIDE" not in out


def test_read_symlink_to_outside_is_refused(tools: RepoTools, repo: Path, outside: Path) -> None:
    os.symlink(outside, repo / "link.txt")
    out = tools.read_file("link.txt")
    assert is_error(out)
    assert "TOP-SECRET-OUTSIDE" not in out


def test_read_through_symlinked_dir_is_refused(tools: RepoTools, repo: Path, outside: Path) -> None:
    os.symlink(outside.parent, repo / "linkdir")
    out = tools.read_file("linkdir/outside.txt")
    assert is_error(out)
    assert "TOP-SECRET-OUTSIDE" not in out


@pytest.mark.parametrize(
    "path", [".git/config", ".git/HEAD", "./.git/config", "src/../.git/config"]
)
def test_read_git_dir_is_refused(tools: RepoTools, path: str) -> None:
    out = tools.read_file(path)
    assert is_error(out)
    assert "[core]" not in out


def test_edit_git_config_is_refused(tools: RepoTools, repo: Path) -> None:
    before = (repo / ".git/config").read_text(encoding="utf-8")
    out = tools.edit_file(".git/config", "[core]", "[core]\n\thooksPath = /tmp/evil")
    assert is_error(out)
    assert (repo / ".git/config").read_text(encoding="utf-8") == before
    assert touched(tools) == set()


def test_create_under_git_dir_is_refused(tools: RepoTools, repo: Path) -> None:
    out = tools.create_file(".git/hooks/pre-commit", "#!/bin/sh\necho pwned\n")
    assert is_error(out)
    assert not (repo / ".git/hooks/pre-commit").exists()


def test_edit_symlink_to_outside_is_refused(tools: RepoTools, repo: Path, outside: Path) -> None:
    os.symlink(outside, repo / "link.txt")
    out = tools.edit_file("link.txt", "TOP-SECRET-OUTSIDE", "changed")
    assert is_error(out)
    assert outside.read_text(encoding="utf-8") == "TOP-SECRET-OUTSIDE\n"
    assert touched(tools) == set()


@pytest.mark.parametrize("path", ["../escaped.txt", "src/../../escaped.txt"])
def test_create_dotdot_escape_is_refused(tools: RepoTools, tmp_path: Path, path: str) -> None:
    out = tools.create_file(path, "pwned\n")
    assert is_error(out)
    assert not (tmp_path / "escaped.txt").exists()
    assert touched(tools) == set()


def test_create_absolute_outside_is_refused(tools: RepoTools, tmp_path: Path) -> None:
    target = tmp_path / "abs-escaped.txt"
    out = tools.create_file(str(target), "pwned\n")
    assert is_error(out)
    assert not target.exists()


def test_create_through_symlinked_dir_is_refused(
    tools: RepoTools, repo: Path, tmp_path: Path
) -> None:
    target_dir = tmp_path / "elsewhere"
    target_dir.mkdir()
    os.symlink(target_dir, repo / "linkdir")
    out = tools.create_file("linkdir/new.txt", "pwned\n")
    assert is_error(out)
    assert not (target_dir / "new.txt").exists()


def test_search_outside_repo_is_refused(tools: RepoTools, outside: Path) -> None:
    out = tools.search_code("TOP-SECRET-OUTSIDE", path="..")
    assert is_error(out)
    assert "outside.txt" not in out


# --------------------------------------------------------------------------- #
# search_code
# --------------------------------------------------------------------------- #


def test_search_code_literal_with_line_numbers(tools: RepoTools, repo: Path) -> None:
    out = tools.search_code("SEEDED BUG")
    lines = [line for line in out.splitlines() if line.startswith(f"{OPS_PY}:")]
    assert len(lines) == 3
    expected = [
        i
        for i, line in enumerate((repo / OPS_PY).read_text(encoding="utf-8").splitlines(), 1)
        if "SEEDED BUG" in line
    ]
    assert [int(line.split(":")[1]) for line in lines] == expected


def test_search_code_is_literal_by_default(tools: RepoTools) -> None:
    out = tools.search_code("a + b")  # '+' would be a regex quantifier
    assert f"{OPS_PY}:" in out
    assert "return a + b" in out


def test_search_code_regex(tools: RepoTools) -> None:
    out = tools.search_code(r"def (subtract|mean)\(", regex=True)
    assert "def subtract(" in out
    assert "def mean(" in out
    assert "def add(" not in out


def test_search_code_path_filter(tools: RepoTools) -> None:
    out = tools.search_code("subtract", path="tests")
    assert "tests/test_ops.py:" in out
    assert f"{OPS_PY}:" not in out


def test_search_code_includes_untracked_files(tools: RepoTools, repo: Path) -> None:
    (repo / "src/calc/fresh.py").write_text("UNIQUE_UNTRACKED_TOKEN = 1\n", encoding="utf-8")
    out = tools.search_code("UNIQUE_UNTRACKED_TOKEN")
    assert "src/calc/fresh.py:1:" in out


def test_search_code_no_match_is_a_string(tools: RepoTools) -> None:
    out = tools.search_code("ZZZ_NOTHING_MATCHES_THIS_ZZZ")
    assert isinstance(out, str)
    assert "ZZZ_NOTHING_MATCHES_THIS_ZZZ:" not in out


def test_search_code_caps_results(tools: RepoTools, repo: Path) -> None:
    (repo / "hits.py").write_text("".join(f"HIT_{i} = {i}\n" for i in range(250)), "utf-8")
    out = tools.search_code("HIT_")
    hits = [line for line in out.splitlines() if re.match(r"^hits\.py:\d+:", line)]
    assert 0 < len(hits) <= 100
    assert MORE_RE.search(out) or "100" in out


def test_search_code_bad_regex_is_error_not_exception(tools: RepoTools) -> None:
    out = tools.search_code("def (unclosed", regex=True)
    assert isinstance(out, str)
    assert is_error(out)


def test_search_code_skips_git_dir(tools: RepoTools) -> None:
    out = tools.search_code("repositoryformatversion")
    assert ".git/" not in out


# --------------------------------------------------------------------------- #
# edit_file
# --------------------------------------------------------------------------- #


def test_edit_file_replaces_exact_match(tools: RepoTools, repo: Path) -> None:
    before = (repo / OPS_PY).read_text(encoding="utf-8")
    out = tools.edit_file(OPS_PY, "return a + b  # SEEDED BUG: should be a - b", "return a - b")
    assert not is_error(out)
    after = (repo / OPS_PY).read_text(encoding="utf-8")
    assert after == before.replace("return a + b  # SEEDED BUG: should be a - b", "return a - b")
    assert touched(tools) == {OPS_PY}


def test_edit_file_no_match_is_error(tools: RepoTools, repo: Path) -> None:
    before = (repo / OPS_PY).read_text(encoding="utf-8")
    out = tools.edit_file(OPS_PY, "this text is not in the file", "x")
    assert is_error(out)
    assert "0" in out or "not found" in out.lower()
    assert (repo / OPS_PY).read_text(encoding="utf-8") == before
    assert touched(tools) == set()


def test_edit_file_multiple_matches_is_error_with_count(tools: RepoTools, repo: Path) -> None:
    before = (repo / OPS_PY).read_text(encoding="utf-8")
    count = before.count("return a + b")
    assert count == 2  # add() and the subtract() bug
    out = tools.edit_file(OPS_PY, "return a + b", "return a - b")
    assert is_error(out)
    assert str(count) in out
    assert (repo / OPS_PY).read_text(encoding="utf-8") == before
    assert touched(tools) == set()


def test_edit_file_is_exact_not_whitespace_insensitive(tools: RepoTools, repo: Path) -> None:
    out = tools.edit_file(OPS_PY, "return  a + b  # SEEDED BUG", "return a - b")
    assert is_error(out)


def test_edit_file_missing_file_is_error(tools: RepoTools, repo: Path) -> None:
    out = tools.edit_file("src/calc/nope.py", "a", "b")
    assert is_error(out)
    assert not (repo / "src/calc/nope.py").exists()


def test_edit_file_can_delete_text(tools: RepoTools, repo: Path) -> None:
    out = tools.edit_file(OPS_PY, "  # SEEDED BUG: should be a - b", "")
    assert not is_error(out)
    assert "SEEDED BUG: should be a - b" not in (repo / OPS_PY).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# create_file
# --------------------------------------------------------------------------- #


def test_create_file_with_parent_dirs(tools: RepoTools, repo: Path) -> None:
    out = tools.create_file("src/calc/sub/helpers.py", "VALUE = 1\n")
    assert not is_error(out)
    assert (repo / "src/calc/sub/helpers.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert touched(tools) == {"src/calc/sub/helpers.py"}


def test_create_existing_file_is_error(tools: RepoTools, repo: Path) -> None:
    before = (repo / OPS_PY).read_text(encoding="utf-8")
    out = tools.create_file(OPS_PY, "overwritten\n")
    assert is_error(out)
    assert (repo / OPS_PY).read_text(encoding="utf-8") == before
    assert touched(tools) == set()


def test_files_touched_accumulates_unique_paths(tools: RepoTools) -> None:
    tools.edit_file(OPS_PY, "  # SEEDED BUG: should be a - b", "")
    tools.edit_file(OPS_PY, "  # SEEDED BUG: off-by-one, should be len(xs)", "")
    tools.create_file("notes/new.txt", "hello\n")
    tools.edit_file(OPS_PY, "not present", "x")  # failed edit does not count
    assert touched(tools) == {OPS_PY, "notes/new.txt"}


def test_files_touched_are_repo_relative(tools: RepoTools, repo: Path) -> None:
    tools.create_file("./a/b.txt", "x\n")
    assert touched(tools) == {"a/b.txt"}


# --------------------------------------------------------------------------- #
# never raises
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "call",
    [
        lambda t: t.read_file(""),
        lambda t: t.read_file("src/calc/ops.py", start_line=0),
        lambda t: t.read_file("src/calc/ops.py", start_line=-5, end_line=-1),
        lambda t: t.read_file("src/calc/ops.py", start_line=10, end_line=2),
        lambda t: t.read_file("bad\x00name"),
        lambda t: t.list_files("bad\x00name"),
        lambda t: t.search_code(""),
        lambda t: t.edit_file("src/calc/ops.py", "", "x"),
        lambda t: t.create_file("", "x"),
        lambda t: t.create_file("src", "x"),
    ],
)
def test_tools_never_raise(tools: RepoTools, call) -> None:
    assert isinstance(call(tools), str)
