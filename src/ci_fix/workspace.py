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
    """A PR checkout inside its run directory.

    Layout: ``run_dir/repo`` (the checkout, ``path``), ``run_dir/venv``, ``run_dir/reports``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_dir: Path
    path: Path
    repo: RepoRef
    pr: PullRequestInfo
    branch: str
    pr_head_sha: str

    @property
    def venv_dir(self) -> Path:
        return self.run_dir / "venv"

    @property
    def reports_dir(self) -> Path:
        return self.run_dir / "reports"


def _require_inside_workspace(path: Path, settings: Settings) -> Path:
    """Resolve ``path``; raise ``ValueError`` unless it is strictly inside ``workspace_dir``."""
    root = settings.workspace_dir.resolve()
    target = Path(path).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError(f"Refusing to delete {path}: not inside workspace dir {root}")
    return target


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

    run_dir = settings.workspace_dir / f"{ref.owner}__{ref.name}__pr-{pr_number}"
    if run_dir.exists():
        target = _require_inside_workspace(run_dir, settings)
        log.info("Removing previous workspace %s", run_dir)
        shutil.rmtree(target)  # the workspace directory is owned by ci-fix
    run_dir.mkdir(parents=True)
    (run_dir / "reports").mkdir()
    dest = run_dir / "repo"

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
    return PreparedRepo(run_dir=run_dir, path=dest, repo=ref, pr=pr, branch=branch, pr_head_sha=sha)


def cleanup_workspace(prepared: PreparedRepo, settings: Settings) -> None:
    """Delete the run directory unless ``settings.keep_workspace`` is set.

    Refuses (``ValueError``) to delete anything that is not strictly inside ``workspace_dir``.
    """
    run_dir = Path(prepared.run_dir)
    if settings.keep_workspace:
        log.info("Keeping workspace at %s", run_dir)
        return
    target = _require_inside_workspace(run_dir, settings)
    shutil.rmtree(target, ignore_errors=True)
    if target.exists():
        log.warning("Could not fully remove workspace %s", run_dir)
        return
    log.info("Removed workspace %s", run_dir)
