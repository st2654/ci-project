"""Thin subprocess wrapper around the git CLI."""

from __future__ import annotations

import base64
import os
import re
import subprocess
import threading
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

from ci_fix.logging_setup import get_logger

log = get_logger(__name__)

_GITHUB_AUTH_HEADER_KEY = "http.https://github.com/.extraheader"
_REDACTED = "***"
# Passed to every git call. The checkout runs untrusted PR code (its tests) before we run git
# in it, so planted hooks or an fsmonitor command must never execute.
_SAFE_CONFIG = ("-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false")
# Never handed to git (or anything it might spawn); auth goes via GIT_CONFIG_* instead.
_SECRET_ENV_KEYS = frozenset({"ANTHROPIC_API_KEY", "GITHUB_TOKEN"})
_SECRET_ENV_PREFIXES = ("CI_FIX_", "PYTHON")
# Identity for local checkpoint commits, so they work without a configured git user.
_COMMIT_CONFIG = (
    "-c",
    "user.name=ci-fix",
    "-c",
    "user.email=ci-fix@localhost",
    "-c",
    "commit.gpgsign=false",
)


class GitError(Exception):
    """Raised when a git command fails or times out."""


def _basic_auth(token: str) -> str:
    return base64.b64encode(f"x-access-token:{token}".encode()).decode()


def _redact(text: str, token: str | None) -> str:
    if not token:
        return text
    for secret in (_basic_auth(token), token):
        text = text.replace(secret, _REDACTED)
    return text


_GLOB_SPECIAL_RE = re.compile(r"([\\*?\[])")
_URL_CREDENTIALS_RE = re.compile(r"(://)[^/@\s]+@")
# Serialises read-dedupe-append of ``info/exclude`` (shared by a repo and its worktrees,
# written from parallel-fix threads).
_EXCLUDE_LOCK = threading.Lock()


def _safe_url(url: str) -> str:
    """Strip any user:password@ part from a URL before logging it."""
    return _URL_CREDENTIALS_RE.sub(r"\1***@", url)


def _git_env(token: str | None) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in _SECRET_ENV_KEYS and not k.startswith(_SECRET_ENV_PREFIXES)
    }
    env["GIT_TERMINAL_PROMPT"] = "0"
    if token:
        # Auth goes through env-scoped config so the token never lands in argv or .git/config.
        env["GIT_CONFIG_COUNT"] = "1"
        env["GIT_CONFIG_KEY_0"] = _GITHUB_AUTH_HEADER_KEY
        env["GIT_CONFIG_VALUE_0"] = f"AUTHORIZATION: basic {_basic_auth(token)}"
    return env


def run_git(
    args: Sequence[str],
    cwd: Path | None = None,
    token: str | None = None,
    timeout: float = 600,
) -> str:
    """Run ``git *args`` and return stdout without the trailing newline."""
    argv = ["git", *_SAFE_CONFIG, *args]
    # The fixed _SAFE_CONFIG flags are left out of the log line for readability.
    shown = " ".join(["git", *args])
    log.debug("$ %s (cwd=%s)", _safe_url(_redact(shown, token)), cwd or ".")
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            env=_git_env(token),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        msg = f"git command timed out after {timeout}s: {argv}\n{stderr.strip()}".rstrip()
        log.debug("git timed out after %.1fs", timeout)
        raise GitError(_redact(msg, token)) from None
    except OSError as exc:
        raise GitError(_redact(f"failed to run {argv}: {exc}", token)) from None

    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        msg = f"git command failed (exit {proc.returncode}): {argv}\n{proc.stderr.strip()}"
        msg = _safe_url(_redact(msg.rstrip(), token))
        log.debug("git exited %d after %.2fs: %s", proc.returncode, elapsed, msg)
        raise GitError(msg)
    log.debug("git ok in %.2fs", elapsed)
    if proc.stderr.strip():
        log.debug("git stderr: %s", _redact(proc.stderr.strip(), token))
    return proc.stdout.rstrip("\n")


def run_git_bytes(args: Sequence[str], cwd: Path | None = None, timeout: float = 600) -> bytes:
    """Run ``git *args`` (no token) and return stdout as raw bytes, untouched.

    For output that must round-trip exactly (patches with CRLF, latin-1 or binary content).
    Only the output's size is logged; stderr is decoded with replacement for errors.
    """
    argv = ["git", *_SAFE_CONFIG, *args]
    log.debug("$ %s (cwd=%s)", " ".join(["git", *args]), cwd or ".")
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv, cwd=cwd, env=_git_env(None), capture_output=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        raise GitError(f"git command timed out after {timeout}s: {argv}") from None
    except OSError as exc:
        raise GitError(f"failed to run {argv}: {exc}") from None
    stderr = proc.stderr.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        raise GitError(f"git command failed (exit {proc.returncode}): {argv}\n{stderr}".rstrip())
    log.debug("git ok in %.2fs (%d bytes)", time.monotonic() - started, len(proc.stdout))
    return proc.stdout


class GitRepo:
    """A local git working copy."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _git(self, *args: str, token: str | None = None) -> str:
        return run_git(args, cwd=self.path, token=token)

    @classmethod
    def clone(cls, url: str, dest: Path, token: str | None = None) -> GitRepo:
        """Clone ``url`` into ``dest``; ``dest`` must not exist or be empty."""
        dest = Path(dest)
        if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
            raise GitError(f"clone destination exists and is not empty: {dest}")
        log.debug("Cloning %s into %s", _safe_url(url), dest)
        started = time.monotonic()
        run_git(["clone", "--no-tags", url, str(dest)], token=token)
        log.debug("Clone finished in %.1fs", time.monotonic() - started)
        return cls(dest)

    def fetch_pr(self, pr_number: int, token: str | None = None) -> str:
        """Fetch the PR head into ``refs/ci-fix-fetch/pr-<N>`` and return its commit SHA.

        The ref namespace differs from the ``ci-fix/pr-<N>`` branch so short names never clash.
        """
        ref = f"refs/ci-fix-fetch/pr-{pr_number}"
        log.debug("Fetching PR #%d head from origin", pr_number)
        self._git("fetch", "origin", f"+pull/{pr_number}/head:{ref}", token=token)
        sha = self._git("rev-parse", ref)
        log.debug("Fetched PR #%d at %s", pr_number, sha[:12])
        return sha

    def fetch_base(self, base_ref: str, token: str | None = None) -> str:
        """Fetch ``base_ref`` from origin into ``refs/ci-fix-fetch/base``; return its SHA."""
        ref = "refs/ci-fix-fetch/base"
        log.debug("Fetching base branch %s from origin", base_ref)
        self._git("fetch", "origin", f"+refs/heads/{base_ref}:{ref}", token=token)
        sha = self._git("rev-parse", ref)
        log.debug("Fetched base %s at %s", base_ref, sha[:12])
        return sha

    def merge_base(self, a: str, b: str) -> str:
        """The best common ancestor of commits ``a`` and ``b``."""
        return self._git("merge-base", a, b)

    def create_branch(self, name: str, start_point: str) -> None:
        self._git("checkout", "-b", name, start_point)
        log.debug("Created branch %s at %s", name, start_point[:12])

    def current_branch(self) -> str:
        """Current branch name, or ``"HEAD"`` when detached."""
        # ``--symbolic-full-name`` stays unambiguous even if another ref shares the short name.
        return self._git("rev-parse", "--symbolic-full-name", "HEAD").removeprefix("refs/heads/")

    def head_sha(self) -> str:
        return self._git("rev-parse", "HEAD")

    def is_clean(self) -> bool:
        return self._git("status", "--porcelain") == ""

    def diff(self, base: str) -> str:
        """Diff of the working tree against ``base``, including uncommitted and new files.

        New (untracked, non-ignored) files are marked with ``git add --intent-to-add`` so they
        show up in the diff; their content is not staged.
        """
        self._git("add", "--all", "--intent-to-add")
        return self._git("diff", base)

    def diff_commits(self, a: str, b: str) -> str:
        """Diff between two commits (``git diff a b``); the working tree is not involved."""
        return self._git("diff", a, b)

    def changed_files(self) -> list[str]:
        """Sorted paths that differ between the working tree and HEAD (new, modified, deleted).

        Gitignored files are not included. A rename is listed as both the old and new path.
        """
        self._git("add", "--all", "--intent-to-add")
        out = self._git("diff", "--name-only", "--no-renames", "HEAD")
        return sorted(set(out.splitlines()))

    def added_files(self) -> set[str]:
        """Paths in the working tree that are not in HEAD (new, non-ignored files)."""
        self._git("add", "--all", "--intent-to-add")
        out = self._git("diff", "--name-only", "--no-renames", "--diff-filter=A", "-z", "HEAD")
        return {p for p in out.split("\0") if p}

    def file_bytes_at(self, rev: str, path: str) -> bytes | None:
        """Raw content of repo-relative ``path`` at commit ``rev`` (None if it is not there)."""
        argv = ["git", *_SAFE_CONFIG, "show", f"{rev}:{path}"]
        log.debug("$ git show %s:%s (cwd=%s)", rev, path, self.path)
        try:
            proc = subprocess.run(
                argv, cwd=self.path, env=_git_env(None), capture_output=True, timeout=600
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.debug("git show failed: %s", exc)
            return None
        return proc.stdout if proc.returncode == 0 else None

    def file_at(self, rev: str, path: str) -> str | None:
        """Text of ``path`` at ``rev`` (invalid UTF-8 replaced); None if it is not there."""
        data = self.file_bytes_at(rev, path)
        return data.decode("utf-8", errors="replace") if data is not None else None

    def files_containing(self, text: str, rev: str = "HEAD", pathspec: str = "*.py") -> list[str]:
        """Paths of files at ``rev`` matching ``pathspec`` that contain ``text`` (fixed string)."""
        try:
            out = self._git("grep", "-l", "-F", "-e", text, rev, "--", pathspec)
        except GitError:  # exit 1 = no match
            return []
        return sorted(line.split(":", 1)[1] for line in out.splitlines() if ":" in line)

    def checkpoint(self, message: str) -> str | None:
        """Commit every change in the working tree; return the new HEAD, or None if unchanged.

        Hooks are skipped (``--no-verify``): these are local bookkeeping commits.
        """
        self._git("add", "--all")
        if not self._git("diff", "--cached", "--name-only"):
            log.debug("Checkpoint skipped: nothing to commit")
            return None
        self._git(*_COMMIT_CONFIG, "commit", "--no-verify", "-q", "-m", message)
        sha = self.head_sha()
        log.debug("Checkpoint %s: %s", sha[:12], message)
        return sha

    def rollback(self) -> None:
        """Discard all uncommitted changes and untracked files (gitignored files are kept)."""
        self.reset_hard()
        self._git("clean", "-fdq")
        log.debug("Rolled back working tree to %s", self.head_sha()[:12])

    def stash_all(self) -> bool:
        """Stash tracked changes and untracked (not ignored) files; False if there were none.

        ``--intent-to-add`` entries (left by ``diff`` and ``changed_files``) make
        ``git stash`` fail, so they are dropped from the index first (the files stay and are
        stashed as untracked). Other staged changes are kept and restored by ``unstash``.
        """
        if self.is_clean():
            log.debug("Stash skipped: nothing to stash")
            return False
        # Index-vs-worktree "added" entries are exactly the intent-to-add ones.
        out = self._git("diff", "--name-only", "--diff-filter=A", "-z")
        intent_to_add = [p for p in out.split("\0") if p]
        if intent_to_add:
            self._git("rm", "--cached", "-q", "--", *intent_to_add)
        self._git(*_COMMIT_CONFIG, "stash", "push", "--include-untracked", "-q", "-m", "ci-fix")
        log.debug("Stashed working tree changes")
        return True

    def unstash(self) -> None:
        """Restore the latest stash (``git stash pop --index``); GitError on a conflict."""
        self._git(*_COMMIT_CONFIG, "stash", "pop", "--index", "-q")
        log.debug("Restored stashed working tree changes")

    def has_stash(self) -> bool:
        """Whether the stash list is non-empty."""
        return self._git("stash", "list") != ""

    def drop_stash(self) -> None:
        """Drop the latest stash entry (GitError if there is none)."""
        self._git("stash", "drop", "-q")
        log.debug("Dropped the latest stash entry")

    def reset_hard(self) -> None:
        """Reset tracked files and the index to HEAD; untracked files are left alone."""
        self._git("reset", "--hard", "-q", "HEAD")

    def untracked_files(self) -> set[str]:
        """Untracked, non-ignored files (one entry per file, also inside untracked dirs)."""
        out = self._git("ls-files", "--others", "--exclude-standard", "-z")
        return {p for p in out.split("\0") if p}

    def exclude(self, paths: Iterable[str]) -> None:
        """Ignore ``paths`` locally via ``.git/info/exclude`` (never committed or cleaned).

        Used for build/test artifacts (``*.egg-info``, ``.coverage``, …) in repos without a
        matching ``.gitignore``, so they never end up in a fix.
        """
        paths = sorted(set(paths))
        if not paths:
            return
        exclude_file = Path(self._git("rev-parse", "--git-path", "info/exclude"))
        if not exclude_file.is_absolute():
            exclude_file = self.path / exclude_file
        exclude_file.parent.mkdir(parents=True, exist_ok=True)
        # Anchored, glob characters escaped: each line matches exactly one path.
        wanted = ["/" + _GLOB_SPECIAL_RE.sub(r"\\\1", p) for p in paths]
        with _EXCLUDE_LOCK:
            existing = exclude_file.read_text(encoding="utf-8") if exclude_file.exists() else ""
            present = set(existing.splitlines())
            new = [line for line in wanted if line not in present]
            if not new:
                return
            lines = "".join(f"{line}\n" for line in new)
            if existing and not existing.endswith("\n"):
                lines = "\n" + lines
            with exclude_file.open("a", encoding="utf-8") as fh:
                fh.write(lines)
        log.debug("Excluded %d untracked artifact(s): %s", len(paths), ", ".join(paths[:10]))

    def add_worktree(self, path: Path, ref: str = "HEAD") -> GitRepo:
        """Check ``ref`` out (detached) into a new linked worktree at ``path``.

        The worktree shares this repo's object store, refs and ``.git/info/exclude``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._git("worktree", "add", "--detach", "-q", str(path), ref)
        log.debug("Added worktree %s at %s", path, ref)
        return GitRepo(path)

    def remove_worktree(self, path: Path) -> None:
        """Remove the linked worktree at ``path`` and prune stale entries; never raises."""
        try:
            self._git("worktree", "remove", "--force", str(path))
        except GitError as exc:
            log.warning("Could not remove worktree %s: %s", path, exc)
        try:
            self._git("worktree", "prune")
        except GitError as exc:
            log.warning("Could not prune worktrees: %s", exc)
        log.debug("Removed worktree %s", path)

    def patch(self, base: str = "HEAD", skip: Iterable[str] = ()) -> bytes:
        """Binary-safe diff of the working tree (new files included) against ``base``.

        Raw bytes, never decoded, so CRLF, non-UTF-8 and binary changes round-trip exactly
        through :meth:`apply_patch`. Gitignored and ``info/exclude``'d files are not included,
        nor are untracked files listed in ``skip`` (e.g. a worktree's test-run artifacts).
        """
        skip = set(skip)
        if not skip:
            self._git("add", "--all", "--intent-to-add")
        else:
            new = sorted(self.untracked_files() - skip)
            if new:
                self._git("add", "--intent-to-add", "--", *(f":(literal){p}" for p in new))
        return run_git_bytes(["diff", "--binary", "--no-color", "--no-ext-diff", base], self.path)

    def apply_patch(self, patch_file: Path) -> None:
        """Apply the patch in ``patch_file`` (raw bytes from :meth:`patch`) to the working tree.

        GitError if it does not apply; nothing is changed then (``git apply`` is atomic).
        """
        self._git("apply", "--whitespace=nowarn", str(patch_file))
        log.debug("Applied patch %s", patch_file)

    def exclude_untracked(self) -> list[str]:
        """Exclude every currently untracked file; return what was excluded."""
        paths = sorted(self.untracked_files())
        self.exclude(paths)
        return paths
