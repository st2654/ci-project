"""Tests for ci_fix.workspace (slice 1): prepare_pr_checkout against a local fake remote."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import REPO_URL, FakeRemote, fake_client, git, pr_info

from ci_fix.config import Settings
from ci_fix.tools.git import GitError
from ci_fix.tools.github import GitHubError, RepoRef
from ci_fix.workspace import PreparedRepo, prepare_pr_checkout


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(workspace_dir=tmp_path / "ws")


def test_prepare_happy_path(fake_remote: FakeRemote, settings: Settings) -> None:
    client = fake_client(pr_info(fake_remote))
    prepared = prepare_pr_checkout(
        REPO_URL, fake_remote.pr_number, settings, client, clone_url=fake_remote.url
    )
    assert isinstance(prepared, PreparedRepo)
    client.get_pull_request.assert_called_once_with(
        RepoRef(owner="octo", name="repo"), fake_remote.pr_number
    )

    expected = settings.workspace_dir / f"octo__repo__pr-{fake_remote.pr_number}"
    assert Path(prepared.path) == expected
    assert prepared.repo == RepoRef(owner="octo", name="repo")
    assert prepared.pr == pr_info(fake_remote)
    assert prepared.branch == f"ci-fix/pr-{fake_remote.pr_number}"
    assert prepared.pr_head_sha == fake_remote.pr_sha

    assert git("symbolic-ref", "HEAD", cwd=expected) == f"refs/heads/{prepared.branch}"
    assert git("rev-parse", "HEAD", cwd=expected) == fake_remote.pr_sha
    assert "a - b" in (expected / "app.py").read_text()
    assert git("status", "--porcelain", cwd=expected) == ""


def test_prepare_uses_branch_prefix(fake_remote: FakeRemote, tmp_path: Path) -> None:
    settings = Settings(workspace_dir=tmp_path / "ws", branch_prefix="fix/")
    prepared = prepare_pr_checkout(
        REPO_URL,
        fake_remote.pr_number,
        settings,
        fake_client(pr_info(fake_remote)),
        clone_url=fake_remote.url,
    )
    assert prepared.branch == f"fix/{fake_remote.pr_number}"
    assert git("symbolic-ref", "HEAD", cwd=Path(prepared.path)) == f"refs/heads/{prepared.branch}"


def test_prepare_rerun_replaces_existing_dest(fake_remote: FakeRemote, settings: Settings) -> None:
    client = fake_client(pr_info(fake_remote))
    first = prepare_pr_checkout(
        REPO_URL, fake_remote.pr_number, settings, client, clone_url=fake_remote.url
    )
    stale = Path(first.path) / "stale.txt"
    stale.write_text("leftover")

    second = prepare_pr_checkout(
        REPO_URL, fake_remote.pr_number, settings, client, clone_url=fake_remote.url
    )
    assert Path(second.path) == Path(first.path)
    assert not stale.exists()
    assert git("rev-parse", "HEAD", cwd=Path(second.path)) == fake_remote.pr_sha


def test_prepare_rerun_over_non_git_dir(fake_remote: FakeRemote, settings: Settings) -> None:
    dest = settings.workspace_dir / f"octo__repo__pr-{fake_remote.pr_number}"
    dest.mkdir(parents=True)
    (dest / "junk.txt").write_text("junk")
    prepared = prepare_pr_checkout(
        REPO_URL,
        fake_remote.pr_number,
        settings,
        fake_client(pr_info(fake_remote)),
        clone_url=fake_remote.url,
    )
    assert not (Path(prepared.path) / "junk.txt").exists()


@pytest.mark.parametrize("state", ["closed", "merged"])
def test_prepare_rejects_non_open_pr(
    fake_remote: FakeRemote, settings: Settings, state: str
) -> None:
    with pytest.raises(GitHubError):
        prepare_pr_checkout(
            REPO_URL,
            fake_remote.pr_number,
            settings,
            fake_client(pr_info(fake_remote, state=state)),
            clone_url=fake_remote.url,
        )
    assert not (settings.workspace_dir / f"octo__repo__pr-{fake_remote.pr_number}").exists()


def test_prepare_rejects_sha_mismatch(fake_remote: FakeRemote, settings: Settings) -> None:
    info = pr_info(fake_remote, head_sha=fake_remote.main_sha)
    with pytest.raises(GitError):
        prepare_pr_checkout(
            REPO_URL, fake_remote.pr_number, settings, fake_client(info), clone_url=fake_remote.url
        )


def test_prepare_invalid_repo_url(fake_remote: FakeRemote, settings: Settings) -> None:
    client = fake_client(pr_info(fake_remote))
    with pytest.raises(GitHubError):
        prepare_pr_checkout(
            "https://gitlab.com/octo/repo",
            fake_remote.pr_number,
            settings,
            client,
            clone_url=fake_remote.url,
        )
    client.get_pull_request.assert_not_called()


def test_prepare_propagates_github_error(fake_remote: FakeRemote, settings: Settings) -> None:
    client = MagicMock()
    client.get_pull_request.side_effect = GitHubError("PR not found")
    with pytest.raises(GitHubError):
        prepare_pr_checkout(
            REPO_URL, fake_remote.pr_number, settings, client, clone_url=fake_remote.url
        )


def test_prepare_missing_pr_ref_raises_git_error(
    fake_remote: FakeRemote, settings: Settings
) -> None:
    info = pr_info(fake_remote).model_copy(update={"number": 999})
    with pytest.raises(GitError):
        prepare_pr_checkout(REPO_URL, 999, settings, fake_client(info), clone_url=fake_remote.url)


def test_prepared_current_branch_not_ambiguous(fake_remote: FakeRemote, settings: Settings) -> None:
    """current_branch returns the plain branch name after a PR checkout."""
    from ci_fix.tools.git import GitRepo

    prepared = prepare_pr_checkout(
        REPO_URL,
        fake_remote.pr_number,
        settings,
        fake_client(pr_info(fake_remote)),
        clone_url=fake_remote.url,
    )
    assert GitRepo(Path(prepared.path)).current_branch() == prepared.branch
