"""Thin subprocess wrapper around the git CLI."""

from __future__ import annotations

import base64
import os
import re
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

from ci_fix.logging_setup import get_logger

log = get_logger(__name__)

_GITHUB_AUTH_HEADER_KEY = "http.https://github.com/.extraheader"
_REDACTED = "***"


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


_URL_CREDENTIALS_RE = re.compile(r"(://)[^/@\s]+@")


def _safe_url(url: str) -> str:
    """Strip any user:password@ part from a URL before logging it."""
    return _URL_CREDENTIALS_RE.sub(r"\1***@", url)


def _git_env(token: str | None) -> dict[str, str]:
    env = dict(os.environ)
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
    argv = ["git", *args]
    log.debug("$ %s (cwd=%s)", _safe_url(_redact(" ".join(argv), token)), cwd or ".")
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
