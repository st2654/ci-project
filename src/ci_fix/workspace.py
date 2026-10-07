"""Prepare a local checkout of a pull request on a fresh ci-fix branch."""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ci_fix.config import Settings
from ci_fix.logging_setup import get_logger
from ci_fix.tools.git import GitError, GitRepo
from ci_fix.tools.github import (
    GitHubClient,
    GitHubError,
    PullRequestInfo,
    RepoRef,
    parse_repo_url,
)

log = get_logger(__name__)


class PreparedRepo(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path
    repo: RepoRef
    pr: PullRequestInfo
    branch: str
    pr_head_sha: str


def prepare_pr_checkout(
    repo_url: str,
    pr_number: int,
    settings: Settings,
    github: GitHubClient,
    clone_url: str | None = None,
) -> PreparedRepo:
    """Clone the repo, fetch the PR head, and create ``<branch_prefix><N>`` from it.

    ``clone_url`` overrides where to clone from (e.g. a local bare repo in tests).
    """
    started = time.monotonic()
    ref = parse_repo_url(repo_url)
    log.info("[setup 1/4] Looking up PR %s#%d", ref.full_name, pr_number)
    pr = github.get_pull_request(ref, pr_number)
    if pr.state != "open":
        raise GitHubError(f"PR {ref.full_name}#{pr_number} is {pr.state}, expected open")

    dest = settings.workspace_dir / f"{ref.owner}__{ref.name}__pr-{pr_number}"
    if dest.exists():
        log.info("Removing previous workspace %s", dest)
        shutil.rmtree(dest)  # the workspace directory is owned by ci-fix
    dest.parent.mkdir(parents=True, exist_ok=True)

    token = settings.github_token.get_secret_value() if settings.github_token else None
    log.info("[setup 2/4] Cloning %s", ref.full_name)
    step = time.monotonic()
    repo = GitRepo.clone(clone_url or ref.clone_url, dest, token=token)
    log.info("[setup 2/4] Cloned in %.1fs", time.monotonic() - step)

    log.info("[setup 3/4] Fetching PR head")
    sha = repo.fetch_pr(pr_number, token=token)
    if sha != pr.head_sha:
        raise GitError(f"PR head moved: API says {pr.head_sha}, fetched {sha} — retry")
    log.info("[setup 3/4] PR head is %s", sha[:12])

    branch = f"{settings.branch_prefix}{pr_number}"
    log.info("[setup 4/4] Creating patch branch %s", branch)
    repo.create_branch(branch, sha)
    log.info("Workspace ready at %s (%.1fs)", dest, time.monotonic() - started)
    return PreparedRepo(path=dest, repo=ref, pr=pr, branch=branch, pr_head_sha=sha)
