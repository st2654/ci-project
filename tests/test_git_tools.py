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


# --------------------------------------------------------------------------- #
# checkpoint / rollback / changed_files
# --------------------------------------------------------------------------- #


def _log(repo: GitRepo) -> list[str]:
    return git("log", "--format=%s|%an|%ae", cwd=repo.path).splitlines()


def test_changed_files_clean_tree(cloned: GitRepo) -> None:
    assert cloned.changed_files() == []


def test_changed_files_new_modified_deleted(cloned: GitRepo) -> None:
    root = Path(cloned.path)
    (root / "app.py").unlink()
    (root / "z_new.py").write_text("x = 1\n")
    (root / "pkg").mkdir()
    (root / "pkg" / "a.py").write_text("y = 2\n")
    assert cloned.changed_files() == ["app.py", "pkg/a.py", "z_new.py"]
    git("checkout", "-q", "HEAD", "--", "app.py", cwd=root)
    (root / "app.py").write_text("changed\n")
    assert cloned.changed_files() == ["app.py", "pkg/a.py", "z_new.py"]


def test_changed_files_ignores_gitignored(cloned: GitRepo) -> None:
    root = Path(cloned.path)
    (root / ".git" / "info" / "exclude").write_text("*.log\n")
    (root / "debug.log").write_text("noise\n")
    assert cloned.changed_files() == []


def test_checkpoint_commits_all_changes(cloned: GitRepo) -> None:
    root = Path(cloned.path)
    before = cloned.head_sha()
    (root / "app.py").write_text("changed\n")
    (root / "new.py").write_text("n = 1\n")
    sha = cloned.checkpoint("ci-fix: fix t (attempt 1)")
    assert sha is not None and sha == cloned.head_sha() != before
    assert cloned.is_clean()
    assert _log(cloned)[0] == "ci-fix: fix t (attempt 1)|ci-fix|ci-fix@localhost"
    assert git("show", "--name-only", "--format=", "HEAD", cwd=root).split() == [
        "app.py",
        "new.py",
    ]


def test_checkpoint_without_changes_returns_none(cloned: GitRepo) -> None:
    before = cloned.head_sha()
    assert cloned.checkpoint("nothing") is None
    assert cloned.head_sha() == before


def test_checkpoint_after_diff_intent_to_add(cloned: GitRepo, fake_remote: FakeRemote) -> None:
    (Path(cloned.path) / "new.py").write_text("n = 1\n")
    cloned.diff(fake_remote.main_sha)  # leaves an intent-to-add entry in the index
    assert cloned.checkpoint("with new file") is not None
    assert "new.py" in git("show", "--name-only", "--format=", "HEAD", cwd=cloned.path)


def test_checkpoint_skips_hooks(cloned: GitRepo) -> None:
    hook = Path(cloned.path) / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    (Path(cloned.path) / "app.py").write_text("changed\n")
    assert cloned.checkpoint("hooks skipped") is not None


def test_rollback_restores_head_and_removes_new_files(cloned: GitRepo) -> None:
    root = Path(cloned.path)
    original = (root / "app.py").read_text()
    (root / "app.py").write_text("changed\n")
    (root / "new.py").write_text("n = 1\n")
    (root / "newdir").mkdir()
    (root / "newdir" / "f.py").write_text("")
    cloned.diff("HEAD")  # intent-to-add entries must not survive a rollback either
    cloned.rollback()
    assert (root / "app.py").read_text() == original
    assert not (root / "new.py").exists()
    assert not (root / "newdir").exists()
    assert cloned.is_clean()


def test_rollback_restores_deleted_file_and_keeps_ignored(cloned: GitRepo) -> None:
    root = Path(cloned.path)
    (root / ".git" / "info" / "exclude").write_text("*.log\n")
    (root / "keep.log").write_text("ignored\n")
    (root / "app.py").unlink()
    cloned.rollback()
    assert (root / "app.py").is_file()
    assert (root / "keep.log").read_text() == "ignored\n"


def test_rollback_keeps_checkpoints(cloned: GitRepo) -> None:
    root = Path(cloned.path)
    (root / "app.py").write_text("accepted\n")
    sha = cloned.checkpoint("accepted")
    (root / "app.py").write_text("rejected\n")
    cloned.rollback()
    assert cloned.head_sha() == sha
    assert (root / "app.py").read_text() == "accepted\n"


# --------------------------------------------------------------------------- #
# untrusted checkout: hooks, fsmonitor and secrets
# --------------------------------------------------------------------------- #

_HOOK_NAMES = (
    "pre-commit",
    "prepare-commit-msg",
    "commit-msg",
    "post-commit",
    "reference-transaction",
)


def _plant_hooks(hooks_dir: Path, marker: Path) -> None:
    hooks_dir.mkdir(parents=True, exist_ok=True)
    for name in _HOOK_NAMES:
        hook = hooks_dir / name
        hook.write_text(f"#!/bin/sh\necho {name} >> '{marker}'\nenv >> '{marker}'\nexit 0\n")
        hook.chmod(0o755)


def _change_and_checkpoint(repo: GitRepo) -> None:
    (Path(repo.path) / "app.py").write_text("changed\n")
    assert repo.checkpoint("checkpoint") is not None
    (Path(repo.path) / "app.py").write_text("again\n")
    repo.changed_files()
    repo.rollback()


def test_planted_repo_hooks_never_run(cloned: GitRepo, tmp_path: Path) -> None:
    marker = tmp_path / "hook-ran"
    _plant_hooks(Path(cloned.path) / ".git" / "hooks", marker)
    _change_and_checkpoint(cloned)
    assert not marker.exists()


def test_planted_hooks_path_in_repo_config_is_ignored(cloned: GitRepo, tmp_path: Path) -> None:
    marker = tmp_path / "hook-ran"
    _plant_hooks(tmp_path / "evil-hooks", marker)
    git("config", "core.hooksPath", str(tmp_path / "evil-hooks"), cwd=cloned.path)
    _change_and_checkpoint(cloned)
    assert not marker.exists()


def test_user_global_hooks_path_is_ignored(
    cloned: GitRepo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "hook-ran"
    _plant_hooks(tmp_path / "global-hooks", marker)
    gitconfig = tmp_path / "gitconfig"
    gitconfig.write_text(f"[core]\n\thooksPath = {tmp_path / 'global-hooks'}\n")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))
    _change_and_checkpoint(cloned)
    assert not marker.exists()


def test_planted_fsmonitor_never_runs(cloned: GitRepo, tmp_path: Path) -> None:
    marker = tmp_path / "fsmonitor-ran"
    script = tmp_path / "fsmonitor.sh"
    script.write_text(f"#!/bin/sh\ntouch '{marker}'\n")
    script.chmod(0o755)
    git("config", "core.fsmonitor", str(script), cwd=cloned.path)
    cloned.is_clean()
    _change_and_checkpoint(cloned)
    assert not marker.exists()


def test_git_subprocess_env_has_no_secrets(
    cloned: GitRepo, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Anything git spawns (a hook, if one ever ran) inherits this env.
    secrets = {
        "GITHUB_TOKEN": "gh-secret-123",
        "ANTHROPIC_API_KEY": "sk-secret-456",
        "CI_FIX_GITHUB_TOKEN": "cfx-secret-789",
        "PYTHONPATH": "/evil/path",
    }
    for key, value in secrets.items():
        monkeypatch.setenv(key, value)
    env_dump = run_git(["-c", "alias.dumpenv=!env", "dumpenv"], cwd=cloned.path)
    for key, value in secrets.items():
        assert key not in env_dump
        assert value not in env_dump
    assert "PATH=" in env_dump  # the rest of the environment is kept


# --------------------------------------------------------------------------- #
# untracked artifacts: untracked_files / exclude
# --------------------------------------------------------------------------- #


def test_untracked_files_lists_files_inside_new_dirs(cloned: GitRepo) -> None:
    root = Path(cloned.path)
    (root / "pkg.egg-info").mkdir()
    (root / "pkg.egg-info" / "PKG-INFO").write_text("x")
    (root / ".coverage").write_text("x")
    (root / "app.py").write_text("tracked change\n")
    assert cloned.untracked_files() == {".coverage", "pkg.egg-info/PKG-INFO"}


def test_exclude_hides_exact_paths_only(cloned: GitRepo) -> None:
    root = Path(cloned.path)
    for name in ("a*b.txt", "aXb.txt", "[x].txt", "x.txt"):
        (root / name).write_text("x")
    (root / "sub").mkdir()
    (root / "sub" / "a*b.txt").write_text("x")
    cloned.exclude(["a*b.txt", "[x].txt"])
    assert cloned.untracked_files() == {"aXb.txt", "x.txt", "sub/a*b.txt"}
    assert cloned.changed_files() == ["aXb.txt", "sub/a*b.txt", "x.txt"]


def test_exclude_untracked_keeps_artifacts_through_checkpoint_and_rollback(
    cloned: GitRepo,
) -> None:
    root = Path(cloned.path)
    (root / ".git" / "info" / "exclude").write_text("# existing, no trailing newline")
    (root / ".coverage").write_text("x")
    assert cloned.exclude_untracked() == [".coverage"]
    assert cloned.changed_files() == []
    assert cloned.checkpoint("nothing") is None
    (root / "app.py").write_text("changed\n")
    cloned.rollback()
    assert (root / ".coverage").is_file()
    exclude = (root / ".git" / "info" / "exclude").read_text()
    assert exclude.splitlines()[-1] == "/.coverage"
