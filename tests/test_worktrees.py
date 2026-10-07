"""Tests for GitRepo.add_worktree / remove_worktree (slice 7)."""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import FakeRemote, git

from ci_fix.tools.git import GitError, GitRepo


@pytest.fixture
def repo(fake_remote: FakeRemote, tmp_path: Path) -> GitRepo:
    repo = GitRepo.clone(fake_remote.url, tmp_path / "clone")
    git("config", "user.email", "tester@example.com", cwd=repo.path)
    git("config", "user.name", "Tester", cwd=repo.path)
    git("config", "commit.gpgsign", "false", cwd=repo.path)
    return repo


def _worktree_paths(repo: GitRepo) -> list[Path]:
    out = git("worktree", "list", "--porcelain", cwd=repo.path)
    return [
        Path(line.split(" ", 1)[1]).resolve()
        for line in out.splitlines()
        if line.startswith("worktree ")
    ]


def test_add_worktree_checks_out_head_detached(repo: GitRepo, tmp_path: Path) -> None:
    # Uncommitted edits in the main checkout must not appear in the worktree.
    (repo.path / "app.py").write_text("dirty\n", encoding="utf-8")
    wt_path = tmp_path / "wt" / "one"

    wt = repo.add_worktree(wt_path)

    assert isinstance(wt, GitRepo)
    assert Path(wt.path).resolve() == wt_path.resolve()
    assert (wt_path / "app.py").read_text(encoding="utf-8") == git(
        "show", "HEAD:app.py", cwd=repo.path
    ) + "\n"
    assert git("rev-parse", "HEAD", cwd=wt_path) == git("rev-parse", "HEAD", cwd=repo.path)
    assert git("rev-parse", "--abbrev-ref", "HEAD", cwd=wt_path) == "HEAD"  # detached
    assert wt_path.resolve() in _worktree_paths(repo)
    # The main checkout keeps its branch and its uncommitted edit.
    assert git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo.path) != "HEAD"
    assert (repo.path / "app.py").read_text(encoding="utf-8") == "dirty\n"


def test_add_worktree_at_explicit_ref(repo: GitRepo, tmp_path: Path) -> None:
    head = git("rev-parse", "HEAD", cwd=repo.path)
    (repo.path / "new.txt").write_text("new\n", encoding="utf-8")
    git("add", "new.txt", cwd=repo.path)
    git("commit", "-q", "-m", "new", cwd=repo.path)

    old = repo.add_worktree(tmp_path / "old", ref=head)
    new = repo.add_worktree(tmp_path / "new")

    assert git("rev-parse", "HEAD", cwd=old.path) == head
    assert not (tmp_path / "old" / "new.txt").exists()
    assert git("rev-parse", "HEAD", cwd=new.path) != head
    assert (tmp_path / "new" / "new.txt").read_text(encoding="utf-8") == "new\n"


def test_worktree_edits_do_not_touch_main_checkout(repo: GitRepo, tmp_path: Path) -> None:
    wt = repo.add_worktree(tmp_path / "wt")
    (Path(wt.path) / "app.py").write_text("changed in worktree\n", encoding="utf-8")

    assert wt.changed_files() == ["app.py"]
    assert repo.changed_files() == []


def test_remove_worktree_cleans_up(repo: GitRepo, tmp_path: Path) -> None:
    wt_path = tmp_path / "wt"
    repo.add_worktree(wt_path)
    (wt_path / "app.py").write_text("dirty\n", encoding="utf-8")  # even when dirty
    (wt_path / "untracked.txt").write_text("x\n", encoding="utf-8")

    repo.remove_worktree(wt_path)

    assert _worktree_paths(repo) == [Path(repo.path).resolve()]
    assert not wt_path.exists()


def test_remove_worktree_twice_and_unknown_path_do_not_raise(repo: GitRepo, tmp_path: Path) -> None:
    wt_path = tmp_path / "wt"
    repo.add_worktree(wt_path)
    repo.remove_worktree(wt_path)
    repo.remove_worktree(wt_path)  # already gone
    repo.remove_worktree(tmp_path / "never-existed")
    assert _worktree_paths(repo) == [Path(repo.path).resolve()]


def test_remove_worktree_whose_directory_was_deleted(repo: GitRepo, tmp_path: Path) -> None:
    import shutil

    wt_path = tmp_path / "wt"
    repo.add_worktree(wt_path)
    shutil.rmtree(wt_path)

    repo.remove_worktree(wt_path)  # must not raise

    git("worktree", "prune", cwd=repo.path)
    assert _worktree_paths(repo) == [Path(repo.path).resolve()]


def test_multiple_worktrees(repo: GitRepo, tmp_path: Path) -> None:
    a = repo.add_worktree(tmp_path / "wts" / "a")
    b = repo.add_worktree(tmp_path / "wts" / "b")
    assert {Path(a.path).resolve(), Path(b.path).resolve()} <= set(_worktree_paths(repo))
    repo.remove_worktree(tmp_path / "wts" / "a")
    repo.remove_worktree(tmp_path / "wts" / "b")
    assert _worktree_paths(repo) == [Path(repo.path).resolve()]


# ---- patch() → apply_patch() round trip ------------------------------------------------------

BASE_FILES: dict[str, bytes] = {
    "mod.py": b"def f():\n    return 1\n",
    "gone.txt": b"to be deleted\n",
    "old_name.txt": b"".join(b"line %d of a file that will be renamed\n" % i for i in range(20)),
    "blob.bin": bytes(range(256)) * 4,
    "crlf.txt": b"first\r\nsecond\r\nthird\r\n",
    "latin.txt": "café crème\n".encode("latin-1"),
    "script.sh": b"echo hi\n",
    "notrail.txt": b"no trailing newline",
}


def _write(root: Path, rel: str, data: bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _snapshot(root: Path, skip: frozenset[str] = frozenset()) -> dict[str, tuple[bytes, bool]]:
    """Every file outside .git: (content, executable bit)."""
    snap = {}
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root).as_posix()
        if ".git" in path.relative_to(root).parts or not path.is_file() or rel in skip:
            continue
        snap[rel] = (path.read_bytes(), bool(path.stat().st_mode & 0o100))
    return snap


@pytest.fixture
def base_repo(repo: GitRepo) -> GitRepo:
    for rel, data in BASE_FILES.items():
        _write(repo.path, rel, data)
    git("add", "-A", cwd=repo.path)
    git("commit", "-q", "-m", "round-trip base", cwd=repo.path)
    return repo


def _edit_modified(wt: Path) -> None:
    _write(wt, "mod.py", b"def f():\n    return 2\n")


def _edit_deleted(wt: Path) -> None:
    (wt / "gone.txt").unlink()


def _edit_renamed(wt: Path) -> None:
    (wt / "pkg").mkdir(exist_ok=True)
    (wt / "old_name.txt").rename(wt / "pkg" / "new_name.txt")


def _edit_binary(wt: Path) -> None:
    _write(wt, "blob.bin", bytes(reversed(range(256))) * 4)
    _write(wt, "new.bin", b"\x00\xff\x00binary\x00")


def _edit_no_trailing_newline(wt: Path) -> None:
    _write(wt, "notrail.txt", b"still no trailing newline, but changed")
    _write(wt, "mod.py", b"def f():\n    return 3")  # newline removed at the end


def _edit_empty_new_file(wt: Path) -> None:
    _write(wt, "pkg/__init__.py", b"")


def _edit_mode(wt: Path) -> None:
    (wt / "script.sh").chmod(0o755)


def _edit_crlf(wt: Path) -> None:
    _write(wt, "crlf.txt", b"first\r\nSECOND\r\nthird\r\n")
    _write(wt, "new_crlf.txt", b"one\r\ntwo\r\n")


def _edit_latin1(wt: Path) -> None:
    _write(wt, "latin.txt", "café crème brûlée\n".encode("latin-1"))
    _write(wt, "new_latin.txt", "naïve\n".encode("latin-1"))


EDITS = {
    "modified": (_edit_modified, ["mod.py"]),
    "deleted": (_edit_deleted, ["gone.txt"]),
    "renamed": (_edit_renamed, ["old_name.txt", "pkg/new_name.txt"]),
    "binary": (_edit_binary, ["blob.bin", "new.bin"]),
    "no_trailing_newline": (_edit_no_trailing_newline, ["mod.py", "notrail.txt"]),
    "empty_new_file": (_edit_empty_new_file, ["pkg/__init__.py"]),
    "chmod_x": (_edit_mode, ["script.sh"]),
    "crlf": (_edit_crlf, ["crlf.txt", "new_crlf.txt"]),
    "latin1": (_edit_latin1, ["latin.txt", "new_latin.txt"]),
}


@pytest.mark.parametrize("name", list(EDITS))
def test_patch_round_trips_exactly(name: str, base_repo: GitRepo, tmp_path: Path) -> None:
    edit, expected_files = EDITS[name]
    wt_path = tmp_path / "wt"
    wt = base_repo.add_worktree(wt_path)
    edit(wt_path)

    patch = wt.patch("HEAD")
    assert isinstance(patch, bytes) and patch
    patch_file = tmp_path / "fix.patch"
    patch_file.write_bytes(patch)
    base_repo.apply_patch(patch_file)

    assert _snapshot(base_repo.path) == _snapshot(wt_path)
    assert base_repo.changed_files() == sorted(expected_files)
    base_repo.remove_worktree(wt_path)


def test_patch_of_unchanged_worktree_is_empty(base_repo: GitRepo, tmp_path: Path) -> None:
    wt = base_repo.add_worktree(tmp_path / "wt")
    assert wt.patch("HEAD") == b""


def test_all_edits_in_one_patch_round_trip(base_repo: GitRepo, tmp_path: Path) -> None:
    wt_path = tmp_path / "wt"
    wt = base_repo.add_worktree(wt_path)
    for edit, _ in EDITS.values():
        if edit is not _edit_no_trailing_newline:  # both touch mod.py; modified wins here
            edit(wt_path)
    patch_file = tmp_path / "fix.patch"
    patch_file.write_bytes(wt.patch("HEAD"))
    base_repo.apply_patch(patch_file)
    assert _snapshot(base_repo.path) == _snapshot(wt_path)


def test_skipped_artifact_is_not_in_patch_but_other_new_files_are(
    base_repo: GitRepo, tmp_path: Path
) -> None:
    wt_path = tmp_path / "wt"
    wt = base_repo.add_worktree(wt_path)
    _edit_modified(wt_path)
    _write(wt_path, "new_module.py", b"X = 1\n")  # the fixer's new file
    _write(wt_path, "pkg/helper.py", b"Y = 2\n")  # another new file, in a new dir
    _write(wt_path, ".coverage", b"artifact")  # test-run artifacts
    _write(wt_path, "out/run-1.log", b"log\n")

    patch = wt.patch("HEAD", skip={".coverage", "out/run-1.log"})
    patch_file = tmp_path / "fix.patch"
    patch_file.write_bytes(patch)
    base_repo.apply_patch(patch_file)

    assert b".coverage" not in patch and b"run-1.log" not in patch
    assert base_repo.changed_files() == ["mod.py", "new_module.py", "pkg/helper.py"]
    skip = frozenset({".coverage", "out/run-1.log"})
    assert _snapshot(base_repo.path) == _snapshot(wt_path, skip)
    # Skipped files stay untracked in the worktree (not staged as intent-to-add).
    assert {".coverage", "out/run-1.log"} <= wt.untracked_files()


def test_skip_handles_special_characters_in_paths(base_repo: GitRepo, tmp_path: Path) -> None:
    wt_path = tmp_path / "wt"
    wt = base_repo.add_worktree(wt_path)
    _write(wt_path, "data/[a]*file?.txt", b"glob chars\n")
    _write(wt_path, "data/artifact.txt", b"artifact\n")

    patch = wt.patch("HEAD", skip={"data/artifact.txt"})
    patch_file = tmp_path / "fix.patch"
    patch_file.write_bytes(patch)
    base_repo.apply_patch(patch_file)

    assert base_repo.changed_files() == ["data/[a]*file?.txt"]


def test_info_excluded_artifact_is_not_in_patch(base_repo: GitRepo, tmp_path: Path) -> None:
    wt_path = tmp_path / "wt"
    wt = base_repo.add_worktree(wt_path)
    _edit_modified(wt_path)
    _write(wt_path, "build.log", b"artifact\n")
    wt.exclude(["build.log"])  # shared info/exclude

    patch = wt.patch("HEAD")
    assert b"build.log" not in patch
    patch_file = tmp_path / "fix.patch"
    patch_file.write_bytes(patch)
    base_repo.apply_patch(patch_file)
    assert base_repo.changed_files() == ["mod.py"]


def test_conflicting_patch_raises_and_leaves_tree_unchanged(
    base_repo: GitRepo, tmp_path: Path
) -> None:
    wt_path = tmp_path / "wt"
    wt = base_repo.add_worktree(wt_path)
    _edit_modified(wt_path)
    _write(wt_path, "brand_new.py", b"Z = 0\n")
    patch_file = tmp_path / "fix.patch"
    patch_file.write_bytes(wt.patch("HEAD"))
    _write(base_repo.path, "mod.py", b"def f():\n    return 99\n")  # conflicting change
    before = _snapshot(base_repo.path)

    with pytest.raises(GitError):
        base_repo.apply_patch(patch_file)
    assert _snapshot(base_repo.path) == before  # git apply is atomic


def test_exclude_is_deduplicated(base_repo: GitRepo) -> None:
    base_repo.exclude(["a.log", "b.log"])
    base_repo.exclude(["b.log", "c.log"])
    base_repo.exclude(["a.log"])
    exclude_file = Path(git("rev-parse", "--git-path", "info/exclude", cwd=base_repo.path))
    if not exclude_file.is_absolute():
        exclude_file = base_repo.path / exclude_file
    lines = [ln for ln in exclude_file.read_text().splitlines() if ln and not ln.startswith("#")]
    assert lines == ["/a.log", "/b.log", "/c.log"]
