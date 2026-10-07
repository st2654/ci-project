"""Tests for ci_fix.tools.github (slice 1): URL parsing, models and GitHubClient (mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock

import github
import pytest
from pydantic import ValidationError

import ci_fix.tools as tools
from ci_fix.tools.github import (
    GitHubClient,
    GitHubError,
    PullRequestInfo,
    RepoRef,
    parse_repo_url,
)

# --------------------------------------------------------------------------- #
# Package surface
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name",
    [
        "GitRepo",
        "GitError",
        "run_git",
        "GitHubClient",
        "GitHubError",
        "RepoRef",
        "PullRequestInfo",
        "parse_repo_url",
    ],
)
def test_tools_package_exports(name: str) -> None:
    assert hasattr(tools, name)


def test_github_error_is_exception() -> None:
    assert issubclass(GitHubError, Exception)


# --------------------------------------------------------------------------- #
# RepoRef / parse_repo_url
# --------------------------------------------------------------------------- #


def test_repo_ref_properties() -> None:
    ref = RepoRef(owner="o", name="r")
    assert ref.full_name == "o/r"
    assert ref.clone_url == "https://github.com/o/r.git"


def test_repo_ref_is_frozen() -> None:
    ref = RepoRef(owner="o", name="r")
    with pytest.raises(ValidationError):
        ref.owner = "x"  # type: ignore[misc]


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/octo/repo",
        "https://github.com/octo/repo.git",
        "https://github.com/octo/repo/",
        "http://github.com/octo/repo",
        "https://www.github.com/octo/repo",
        "git@github.com:octo/repo",
        "git@github.com:octo/repo.git",
        "  https://github.com/octo/repo  \n",
    ],
)
def test_parse_repo_url_valid(url: str) -> None:
    assert parse_repo_url(url) == RepoRef(owner="octo", name="repo")


def test_parse_repo_url_keeps_dots_and_dashes_in_name() -> None:
    assert parse_repo_url("https://github.com/my-org/my.repo-x") == RepoRef(
        owner="my-org", name="my.repo-x"
    )


@pytest.mark.parametrize(
    "url",
    [
        "",
        "not a url",
        "https://gitlab.com/octo/repo",
        "https://github.example.com/octo/repo",
        "https://github.com/octo",
        "https://github.com/octo/",
        "https://github.com/",
        "https://github.com/octo/repo/pull/3",
        "https://github.com/octo/repo/tree/main",
        "git@gitlab.com:octo/repo.git",
    ],
)
def test_parse_repo_url_invalid(url: str) -> None:
    with pytest.raises(GitHubError):
        parse_repo_url(url)


# --------------------------------------------------------------------------- #
# GitHubClient.get_pull_request (PyGithub mocked)
# --------------------------------------------------------------------------- #


def make_pr(
    *,
    number: int = 7,
    state: str = "open",
    base_repo: str = "octo/repo",
    head_repo: str | None = "octo/repo",
    maintainer_can_modify: bool = False,
) -> MagicMock:
    pr = MagicMock()
    pr.number = number
    pr.title = "Fix the thing"
    pr.state = state
    pr.base.ref = "main"
    pr.base.repo.full_name = base_repo
    pr.head.ref = "feature"
    pr.head.sha = "a" * 40
    if head_repo is None:
        pr.head.repo = None
    else:
        pr.head.repo.full_name = head_repo
        pr.head.repo.clone_url = f"https://github.com/{head_repo}.git"
    pr.maintainer_can_modify = maintainer_can_modify
    pr.html_url = f"https://github.com/{base_repo}/pull/{number}"
    return pr


def make_client(pr: MagicMock) -> tuple[GitHubClient, MagicMock]:
    gh = MagicMock()
    gh.get_repo.return_value.get_pull.return_value = pr
    return GitHubClient(github=gh), gh


REPO = RepoRef(owner="octo", name="repo")


def test_get_pull_request_maps_fields() -> None:
    client, gh = make_client(make_pr())
    info = client.get_pull_request(REPO, 7)
    gh.get_repo.assert_called_once_with("octo/repo")
    gh.get_repo.return_value.get_pull.assert_called_once_with(7)
    assert isinstance(info, PullRequestInfo)
    assert info.number == 7
    assert info.title == "Fix the thing"
    assert info.state == "open"
    assert info.base_ref == "main"
    assert info.head_ref == "feature"
    assert info.head_sha == "a" * 40
    assert info.head_repo_full_name == "octo/repo"
    assert info.is_fork is False
    assert info.html_url == "https://github.com/octo/repo/pull/7"


def test_get_pull_request_fork() -> None:
    client, _ = make_client(make_pr(head_repo="someone/repo"))
    info = client.get_pull_request(REPO, 7)
    assert info.is_fork is True
    assert info.head_repo_full_name == "someone/repo"


def test_get_pull_request_deleted_head_repo_is_fork() -> None:
    client, _ = make_client(make_pr(head_repo=None))
    info = client.get_pull_request(REPO, 7)
    assert info.is_fork is True
    assert info.head_repo_full_name is None


def test_pull_request_info_is_frozen() -> None:
    client, _ = make_client(make_pr())
    info = client.get_pull_request(REPO, 7)
    with pytest.raises(ValidationError):
        info.state = "closed"  # type: ignore[misc]


def test_get_pull_request_404_raises_not_found() -> None:
    gh = MagicMock()
    gh.get_repo.return_value.get_pull.side_effect = github.GithubException(
        404, {"message": "Not Found"}, None
    )
    client = GitHubClient(github=gh)
    with pytest.raises(GitHubError) as exc:
        client.get_pull_request(REPO, 7)
    assert "not found" in str(exc.value).lower()


def test_get_repo_404_raises_github_error() -> None:
    gh = MagicMock()
    gh.get_repo.side_effect = github.GithubException(404, {"message": "Not Found"}, None)
    client = GitHubClient(github=gh)
    with pytest.raises(GitHubError) as exc:
        client.get_pull_request(REPO, 7)
    assert "not found" in str(exc.value).lower()


def test_get_pull_request_other_error_raises_github_error() -> None:
    gh = MagicMock()
    gh.get_repo.return_value.get_pull.side_effect = github.GithubException(
        500, {"message": "Server Error"}, None
    )
    with pytest.raises(GitHubError):
        GitHubClient(github=gh).get_pull_request(REPO, 7)


def test_client_constructs_without_args() -> None:
    GitHubClient()  # must not hit the network on construction


def test_get_pull_request_fork_push_fields() -> None:
    client, _ = make_client(make_pr(head_repo="someone/repo", maintainer_can_modify=True))
    info = client.get_pull_request(RepoRef(owner="octo", name="repo"), 7)
    assert info.head_clone_url == "https://github.com/someone/repo.git"
    assert info.maintainer_can_modify is True


def test_get_pull_request_deleted_fork_has_no_clone_url() -> None:
    client, _ = make_client(make_pr(head_repo=None))
    info = client.get_pull_request(RepoRef(owner="octo", name="repo"), 7)
    assert info.head_clone_url is None
    assert info.maintainer_can_modify is False
