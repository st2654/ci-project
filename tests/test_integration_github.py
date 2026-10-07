"""Integration test (slice 1): real GitHub PR checkout. Opt-in via CI_FIX_RUN_INTEGRATION=1."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from conftest import git

from ci_fix.config import Settings
from ci_fix.tools.github import GitHubClient, GitHubError, parse_repo_url
from ci_fix.workspace import prepare_pr_checkout

REPO_URL = "https://github.com/octocat/Hello-World"
PR_NUMBER = 32  # long-lived open PR (since 2012)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("CI_FIX_RUN_INTEGRATION") != "1",
        reason="set CI_FIX_RUN_INTEGRATION=1 to run against real GitHub",
    ),
]


def test_prepare_real_public_pr(tmp_path: Path) -> None:
    client = GitHubClient(token=os.environ.get("GITHUB_TOKEN") or None)
    pr = client.get_pull_request(parse_repo_url(REPO_URL), PR_NUMBER)
    if pr.state != "open":
        pytest.skip(f"{REPO_URL} PR #{PR_NUMBER} is {pr.state!r}, not open")

    try:
        prepared = prepare_pr_checkout(
            REPO_URL, PR_NUMBER, Settings(workspace_dir=tmp_path), client
        )
    except GitHubError as exc:  # e.g. PR closed between the two calls
        pytest.skip(f"PR no longer usable: {exc}")

    path = Path(prepared.path)
    assert prepared.branch == f"ci-fix/pr-{PR_NUMBER}"
    assert git("symbolic-ref", "HEAD", cwd=path) == f"refs/heads/ci-fix/pr-{PR_NUMBER}"
    assert git("rev-parse", "HEAD", cwd=path) == prepared.pr.head_sha
    assert prepared.pr_head_sha == prepared.pr.head_sha
