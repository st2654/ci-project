"""Tests for ci_fix.tools.git (slice 1): run_git and GitRepo against a local fake remote."""

from __future__ import annotations

import base64
import subprocess
from pathlib import Path

import pytest
from conftest import FakeRemote, git

from ci_fix.tools.git import GitError, GitRepo, run_git

TOKEN = "s3cr3t-tok"


@pytest.fixture
def cloned(fake_remote: FakeRemote, tmp_path: Path) -> GitRepo:
    return GitRepo.clone(fake_remote.url, tmp_path / "clone")


# --------------------------------------------------------------------------- #
# run_git
# --------------------------------------------------------------------------- #


def test_git_error_is_exception() -> None:
    assert issubclass(GitError, Exception)


def test_run_git_returns_stdout_without_trailing_newline(fake_remote: FakeRemote) -> None:
    out = run_git(["rev-parse", "main"], cwd=fake_remote.bare)
    assert out == fake_remote.main_sha
    assert not out.endswith("\n")


def test_run_git_nonzero_exit_raises_with_stderr(fake_remote: FakeRemote) -> None:
    with pytest.raises(GitError) as exc:
        run_git(["rev-parse", "--verify", "no-such-ref-xyz"], cwd=fake_remote.bare)
    # git prints "fatal: Needed a single revision" on stderr
    assert "fatal" in str(exc.value).lower() or "single revision" in str(exc.value)


def test_run_git_redacts_token_in_error(cloned: GitRepo) -> None:
    with pytest.raises(GitError) as exc:
        # stderr will echo the bad ref name, which equals the token
        run_git(["checkout", TOKEN], cwd=cloned.path, token=TOKEN)
    msg = str(exc.value)
    assert TOKEN not in msg
    assert "***" in msg
    assert all(TOKEN not in str(a) for a in exc.value.args)


def test_run_git_token_passed_via_env_config(cloned: GitRepo) -> None:
    out = run_git(
        ["config", "--get-regexp", r"^http\..*extraheader$"], cwd=cloned.path, token=TOKEN
    )
    assert "authorization" in out.lower()
    assert TOKEN not in (cloned.path / ".git" / "config").read_text()


def test_run_git_token_never_in_argv(cloned: GitRepo, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []
    real_run = subprocess.run
    real_popen = subprocess.Popen

    def spy_run(args, *a, **kw):  # type: ignore[no-untyped-def]
        calls.append([str(x) for x in args])
        return real_run(args, *a, **kw)

    class SpyPopen(real_popen):  # type: ignore[misc,valid-type]
        def __init__(self, args, *a, **kw):  # type: ignore[no-untyped-def]
            calls.append([str(x) for x in args])
            super().__init__(args, *a, **kw)

    monkeypatch.setattr(subprocess, "run", spy_run)
    monkeypatch.setattr(subprocess, "Popen", SpyPopen)
    run_git(["status", "--porcelain"], cwd=cloned.path, token=TOKEN)
    if not calls:
        pytest.skip("run_git does not use subprocess.run/Popen; argv not observable")
    b64 = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()
    for argv in calls:
        joined = " ".join(argv)
        assert TOKEN not in joined
        assert b64 not in joined


# --------------------------------------------------------------------------- #
# GitRepo.clone
# --------------------------------------------------------------------------- #


def test_clone_from_local_path(fake_remote: FakeRemote, tmp_path: Path) -> None:
    dest = tmp_path / "c1"
    repo = GitRepo.clone(fake_remote.url, dest)
    assert isinstance(repo, GitRepo)
    assert Path(repo.path) == dest
    assert (dest / "app.py").is_file()
    assert repo.head_sha() == fake_remote.main_sha
    assert repo.current_branch() == "main"


def test_clone_from_file_url(fake_remote: FakeRemote, tmp_path: Path) -> None:
    repo = GitRepo.clone(fake_remote.bare.as_uri(), tmp_path / "c2")
    assert repo.head_sha() == fake_remote.main_sha


def test_clone_into_non_empty_dest_raises(fake_remote: FakeRemote, tmp_path: Path) -> None:
    dest = tmp_path / "busy"
    dest.mkdir()
    (dest / "keep.txt").write_text("x")
    with pytest.raises(GitError):
        GitRepo.clone(fake_remote.url, dest)
    assert (dest / "keep.txt").read_text() == "x"


def test_clone_bad_url_raises(tmp_path: Path) -> None:
    with pytest.raises(GitError):
        GitRepo.clone(str(tmp_path / "does-not-exist.git"), tmp_path / "c3")


def test_clone_with_token_does_not_persist_token(fake_remote: FakeRemote, tmp_path: Path) -> None:
    repo = GitRepo.clone(fake_remote.url, tmp_path / "c4", token=TOKEN)
    config = (Path(repo.path) / ".git" / "config").read_text()
    b64 = base64.b64encode(f"x-access-token:{TOKEN}".encode()).decode()
    assert TOKEN not in config
    assert b64 not in config
    assert "extraheader" not in config.lower()


# --------------------------------------------------------------------------- #
# GitRepo operations
# --------------------------------------------------------------------------- #


def test_fetch_pr_returns_sha_and_creates_ref(cloned: GitRepo, fake_remote: FakeRemote) -> None:
    sha = cloned.fetch_pr(fake_remote.pr_number)
    assert sha == fake_remote.pr_sha
    assert git("rev-parse", f"refs/ci-fix-fetch/pr-{fake_remote.pr_number}", cwd=cloned.path) == sha


def test_fetch_pr_with_token(cloned: GitRepo, fake_remote: FakeRemote) -> None:
    assert cloned.fetch_pr(fake_remote.pr_number, token=TOKEN) == fake_remote.pr_sha
    assert TOKEN not in (Path(cloned.path) / ".git" / "config").read_text()


def test_fetch_pr_missing_ref_raises(cloned: GitRepo) -> None:
    with pytest.raises(GitError):
        cloned.fetch_pr(999)


def test_create_branch_at_start_point(cloned: GitRepo, fake_remote: FakeRemote) -> None:
    sha = cloned.fetch_pr(fake_remote.pr_number)
    cloned.create_branch("ci-fix/pr-7", sha)
    assert git("rev-parse", "refs/heads/ci-fix/pr-7", cwd=cloned.path) == sha


def test_create_existing_branch_raises(cloned: GitRepo, fake_remote: FakeRemote) -> None:
    with pytest.raises(GitError):
        cloned.create_branch("main", fake_remote.main_sha)


def test_head_sha_and_current_branch_follow_checkout(
    cloned: GitRepo, fake_remote: FakeRemote
) -> None:
    sha = cloned.fetch_pr(fake_remote.pr_number)
    cloned.create_branch("work", sha)
    git("checkout", "-q", "work", cwd=cloned.path)
    assert cloned.current_branch() == "work"
    assert cloned.head_sha() == fake_remote.pr_sha


def test_is_clean(cloned: GitRepo) -> None:
    assert cloned.is_clean() is True
    (Path(cloned.path) / "app.py").write_text("changed\n")
    assert cloned.is_clean() is False


def test_diff_includes_uncommitted_changes(cloned: GitRepo, fake_remote: FakeRemote) -> None:
    assert cloned.diff(fake_remote.main_sha) == ""
    (Path(cloned.path) / "app.py").write_text("def add(a, b):\n    return a * b\n")
    d = cloned.diff(fake_remote.main_sha)
    assert "app.py" in d
    assert "-    return a + b" in d
    assert "+    return a * b" in d


def test_diff_includes_committed_changes_vs_base(cloned: GitRepo, fake_remote: FakeRemote) -> None:
    sha = cloned.fetch_pr(fake_remote.pr_number)
    cloned.create_branch("work", sha)
    git("checkout", "-q", "work", cwd=cloned.path)
    d = cloned.diff(fake_remote.main_sha)
    assert "+    return a - b" in d


def test_diff_includes_new_untracked_files(cloned: GitRepo, fake_remote: FakeRemote) -> None:
    (Path(cloned.path) / "new_module.py").write_text("VALUE = 1\n")
    d = cloned.diff(fake_remote.main_sha)
    assert "new_module.py" in d
    assert "+VALUE = 1" in d


def test_diff_excludes_gitignored_files(cloned: GitRepo, fake_remote: FakeRemote) -> None:
    (Path(cloned.path) / ".gitignore").write_text("*.log\n")
    (Path(cloned.path) / "debug.log").write_text("noise\n")
    d = cloned.diff(fake_remote.main_sha)
    assert "debug.log" not in d.replace(".gitignore", "")
