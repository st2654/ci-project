"""GitHub API access (PyGithub) and repository URL parsing."""

from __future__ import annotations

import re

from github import Auth, Github, GithubException
from pydantic import BaseModel

from ci_fix.logging_setup import get_logger

log = get_logger(__name__)

_NAME = r"[A-Za-z0-9_.-]+"
_REPO_URL_RE = re.compile(
    rf"^(?:https?://(?:www\.)?github\.com/|git@github\.com:)(?P<owner>{_NAME})/(?P<name>{_NAME})/?$"
)


class GitHubError(Exception):
    """Raised for GitHub API failures and invalid repository references."""


class RepoRef(BaseModel, frozen=True):
    owner: str
    name: str

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.name}"

    @property
    def clone_url(self) -> str:
        return f"https://github.com/{self.full_name}.git"


def parse_repo_url(url: str) -> RepoRef:
    """Parse a github.com HTTPS or SSH repository URL into a RepoRef."""
    match = _REPO_URL_RE.match(url.strip())
    if match is None:
        raise GitHubError(f"Not a GitHub repository URL: {url!r}")
    owner, name = match["owner"], match["name"]
    name = name.removesuffix(".git")
    if not name or name in {".", ".."} or owner in {".", ".."}:
        raise GitHubError(f"Not a GitHub repository URL: {url!r}")
    return RepoRef(owner=owner, name=name)


class PullRequestInfo(BaseModel, frozen=True):
    number: int
    title: str
    state: str
    base_ref: str
    head_ref: str
    head_sha: str
    head_repo_full_name: str | None
    is_fork: bool
    html_url: str
    # Needed to push a fix to a fork PR's branch (allowed only if maintainer_can_modify).
    head_clone_url: str | None = None
    maintainer_can_modify: bool = False
    body: str = ""


class GitHubClient:
    """Small facade over PyGithub returning plain models."""

    def __init__(self, token: str | None = None, github: Github | None = None) -> None:
        if github is None:
            github = Github(auth=Auth.Token(token)) if token else Github()
        self._github = github

    def get_pull_request(self, repo: RepoRef, number: int) -> PullRequestInfo:
        label = f"{repo.full_name}#{number}"
        log.debug("Fetching pull request %s", label)
        try:
            pr = self._github.get_repo(repo.full_name).get_pull(number)
            head_repo = pr.head.repo
            head_full_name = head_repo.full_name if head_repo is not None else None
            info = PullRequestInfo(
                number=pr.number,
                title=pr.title,
                state=pr.state,
                base_ref=pr.base.ref,
                head_ref=pr.head.ref,
                head_sha=pr.head.sha,
                head_repo_full_name=head_full_name,
                is_fork=head_full_name != pr.base.repo.full_name,
                html_url=pr.html_url,
                head_clone_url=head_repo.clone_url if head_repo is not None else None,
                maintainer_can_modify=pr.maintainer_can_modify is True,
                body=pr.body if isinstance(pr.body, str) else "",
            )
        except GithubException as exc:
            if exc.status == 404:
                raise GitHubError(f"Pull request {label} not found or not accessible") from exc
            raise GitHubError(
                f"GitHub API error fetching pull request {label} (HTTP {exc.status}): {exc.data}"
            ) from exc
        log.info(
            "PR %s %r: %s -> %s, state=%s, head=%s%s",
            label,
            info.title,
            info.head_ref,
            info.base_ref,
            info.state,
            info.head_sha[:12],
            f", fork={info.head_repo_full_name or '<deleted>'}" if info.is_fork else "",
        )
        return info
