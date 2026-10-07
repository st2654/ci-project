"""Shared fixtures: a fake "GitHub" remote built from local git repos (no network)."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ci_fix.tools.github import PullRequestInfo

PR_NUMBER = 7


def git(*args: str, cwd: Path | None = None) -> str:
    """Run git directly (independent of the code under test) and return stdout."""
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }
    result = subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


@dataclass(frozen=True)
class FakeRemote:
    """A bare repo standing in for GitHub, with ``refs/pull/<N>/head`` set."""

    bare: Path
    main_sha: str
    pr_sha: str
    pr_number: int = PR_NUMBER

    @property
    def url(self) -> str:
        return str(self.bare)


@pytest.fixture
def fake_remote(tmp_path: Path) -> FakeRemote:
    work = tmp_path / "origin-work"
    work.mkdir()
    git("init", "-q", "-b", "main", cwd=work)
    git("config", "user.email", "tester@example.com", cwd=work)
    git("config", "user.name", "Tester", cwd=work)
    git("config", "commit.gpgsign", "false", cwd=work)

    (work / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    git("add", "app.py", cwd=work)
    git("commit", "-q", "-m", "initial", cwd=work)
    main_sha = git("rev-parse", "HEAD", cwd=work)

    git("checkout", "-q", "-b", "feature", cwd=work)
    (work / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    git("commit", "-q", "-am", "feature change", cwd=work)
    pr_sha = git("rev-parse", "HEAD", cwd=work)
    git("checkout", "-q", "main", cwd=work)

    bare = tmp_path / "remote.git"
    git("clone", "-q", "--bare", str(work), str(bare))
    git("update-ref", f"refs/pull/{PR_NUMBER}/head", pr_sha, cwd=bare)
    return FakeRemote(bare=bare, main_sha=main_sha, pr_sha=pr_sha)


# Shared helpers for workspace-level tests.
REPO_URL = "https://github.com/octo/repo"


def pr_info(remote: FakeRemote, *, state: str = "open", head_sha: str | None = None):
    return PullRequestInfo(
        number=remote.pr_number,
        title="Feature",
        state=state,
        base_ref="main",
        head_ref="feature",
        head_sha=head_sha or remote.pr_sha,
        head_repo_full_name="octo/repo",
        is_fork=False,
        html_url=f"{REPO_URL}/pull/{remote.pr_number}",
    )


def fake_client(info: PullRequestInfo) -> MagicMock:
    client = MagicMock()
    client.get_pull_request.return_value = info
    return client
