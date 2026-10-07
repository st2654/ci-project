"""Delivery tests (slice 8): squash commit, push, fix PR, comment on the original PR.

Each test builds its OWN sample remote (a local bare repo) because delivery pushes to it;
the session-wide ``sample_remote`` must never be written to. GitHub is an in-memory fake
implementing the ``GitHubClient`` delivery methods (or the real client over a mocked PyGithub).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from conftest import git
from github import GithubException
from pipeline_helpers import (
    DIV_ZERO,
    DIVIDE_FIX,
    FIX_DIVIDE,
    FIX_SUBTRACT,
    OPS_PY,
    REPO_URL,
    SAMPLE_PR,
    SUBTRACT,
    SUBTRACT_FIX,
    FakeRunnerFactory,
    SampleRemote,
    ScriptedFixer,
    make_deps,
    make_sample_remote,
    run_dirs_for,
    sample_pr_info,
)

from ci_fix.models import FixResult, OutcomeStatus
from ci_fix.pipeline import fix_failing_tests
from ci_fix.report import COMMENT_MARKER
from ci_fix.tools.git import GitError
from ci_fix.tools.github import GitHubClient, GitHubError, PullRef, PullRequestInfo, RepoRef

BRANCH = f"ci-fix/pr-{SAMPLE_PR}"
FAKE_TOKEN = "ghp_FAKEtoken0123456789abcdefSECRET"
WRITE_METHODS = {"create_pull", "edit_pull", "create_comment", "edit_comment"}


# ---- fakes ----------------------------------------------------------------------------------


@dataclass
class FakeGitHub:
    """In-memory GitHub: remembers PRs and comments so re-runs see earlier deliveries."""

    info: PullRequestInfo
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    pulls: dict[int, dict[str, Any]] = field(default_factory=dict)
    comments: dict[int, dict[str, Any]] = field(default_factory=dict)
    fail: dict[str, Exception] = field(default_factory=dict)
    next_pr: int = 100
    default_branch: str = "main"
    # Head sha the API reports after the first lookup (simulates a push to the PR mid-run).
    moved_head: str | None = None
    pr_lookups: int = 0

    def _record(self, name: str, **kw: Any) -> None:
        self.calls.append((name, kw))
        if name in self.fail:
            raise self.fail[name]

    def calls_to(self, name: str) -> list[dict[str, Any]]:
        return [kw for n, kw in self.calls if n == name]

    @property
    def writes(self) -> list[str]:
        return [n for n, _ in self.calls if n in WRITE_METHODS]

    def get_pull_request(self, repo: RepoRef, number: int) -> PullRequestInfo:
        self.pr_lookups += 1
        if self.moved_head and self.pr_lookups > 1:
            return self.info.model_copy(update={"head_sha": self.moved_head})
        return self.info

    def get_default_branch(self, repo: RepoRef) -> str:
        self._record("get_default_branch", repo=repo)
        return self.default_branch

    def find_open_pr(self, repo: RepoRef, head: str, base: str) -> PullRef | None:
        self._record("find_open_pr", repo=repo, head=head, base=base)
        branch = head.split(":", 1)[-1]
        for number, pr in self.pulls.items():
            if pr["repo"] == repo and pr["head"] == branch and pr["base"] == base:
                return PullRef(number=number, html_url=pr["url"])
        return None

    def create_pull(self, repo: RepoRef, title: str, body: str, head: str, base: str) -> PullRef:
        self._record("create_pull", repo=repo, title=title, body=body, head=head, base=base)
        number = self.next_pr
        self.next_pr += 1
        url = f"https://github.com/{repo.full_name}/pull/{number}"
        self.pulls[number] = {
            "repo": repo,
            "head": head.split(":", 1)[-1],
            "base": base,
            "title": title,
            "body": body,
            "url": url,
        }
        return PullRef(number=number, html_url=url)

    def edit_pull(self, repo: RepoRef, number: int, title: str, body: str) -> PullRef:
        self._record("edit_pull", repo=repo, number=number, title=title, body=body)
        self.pulls[number].update(title=title, body=body)
        return PullRef(number=number, html_url=self.pulls[number]["url"])

    def find_comment(self, repo: RepoRef, number: int, marker: str) -> int | None:
        self._record("find_comment", repo=repo, number=number, marker=marker)
        for cid, c in self.comments.items():
            if c["number"] == number and marker in c["body"]:
                return cid
        return None

    def create_comment(self, repo: RepoRef, number: int, body: str) -> str:
        self._record("create_comment", repo=repo, number=number, body=body)
        cid = 5000 + len(self.comments)
        self.comments[cid] = {"number": number, "body": body}
        return f"https://github.com/{repo.full_name}/pull/{number}#issuecomment-{cid}"

    def edit_comment(self, repo: RepoRef, number: int, comment_id: int, body: str) -> str:
        self._record("edit_comment", repo=repo, number=number, comment_id=comment_id, body=body)
        self.comments[comment_id]["body"] = body
        return f"https://github.com/{repo.full_name}/pull/{number}#issuecomment-{comment_id}"


# ---- harness --------------------------------------------------------------------------------


@pytest.fixture
def remote(tmp_path: Path) -> SampleRemote:
    """A private, writable sample remote for this test."""
    return make_sample_remote(tmp_path / "remote")


def run_pipeline(
    tmp_path: Path,
    remote: SampleRemote,
    fixer: Any,
    tests: list[str],
    github: Any,
    **settings: Any,
) -> FixResult:
    settings.setdefault("push", True)  # make_deps defaults to a dry run
    deps = make_deps(tmp_path, fixer, remote=remote, runner_factory=FakeRunnerFactory(), **settings)
    deps = dataclasses.replace(deps, github=github)
    return fix_failing_tests(REPO_URL, SAMPLE_PR, tests, deps=deps)


def remote_branches(bare: Path) -> list[str]:
    out = git("for-each-ref", "--format=%(refname:short)", "refs/heads", cwd=bare)
    return out.splitlines()


def commits_on_top(bare: Path, base: str, branch: str = BRANCH) -> list[str]:
    return git("rev-list", f"{base}..{branch}", cwd=bare).splitlines()


def two_fixes() -> ScriptedFixer:
    return ScriptedFixer({SUBTRACT: [FIX_SUBTRACT], DIV_ZERO: [FIX_DIVIDE]})


def no_fixes() -> ScriptedFixer:
    return ScriptedFixer({SUBTRACT: [("unfixable", "needs product input")]})


@pytest.fixture
def gh(remote: SampleRemote) -> FakeGitHub:
    return FakeGitHub(sample_pr_info(remote))


# ---- happy path -----------------------------------------------------------------------------


def test_two_fixes_one_commit_pushed_pr_and_comment(
    tmp_path: Path, remote: SampleRemote, gh: FakeGitHub
) -> None:
    result = run_pipeline(tmp_path, remote, two_fixes(), [SUBTRACT, DIV_ZERO], gh)

    assert [t.status for t in result.tests] == [OutcomeStatus.FIXED, OutcomeStatus.FIXED]
    assert BRANCH in remote_branches(remote.bare)
    commits = commits_on_top(remote.bare, remote.pr_sha)
    assert len(commits) == 1
    assert git("rev-parse", f"{BRANCH}^", cwd=remote.bare) == remote.pr_sha
    assert result.commit_sha == commits[0]

    author = git("log", "-1", "--format=%an <%ae>|%cn <%ce>", BRANCH, cwd=remote.bare)
    assert (
        author
        == "ci-fix <ci-fix@users.noreply.github.com>|ci-fix <ci-fix@users.noreply.github.com>"
    )
    subject = git("log", "-1", "--format=%s", BRANCH, cwd=remote.bare)
    assert subject == f"ci-fix: fix 2 failing tests in #{SAMPLE_PR}"
    full_message = git("log", "-1", "--format=%B", BRANCH, cwd=remote.bare)
    assert full_message.strip() == result.commit_message.strip()

    ops = git("show", f"{BRANCH}:{OPS_PY}", cwd=remote.bare)
    assert SUBTRACT_FIX in ops
    assert DIVIDE_FIX in ops
    # The PR's own change is still there (the commit sits on the PR head).
    assert git("show", f"{BRANCH}:PR_CHANGE.txt", cwd=remote.bare) == "change from the PR"

    (create,) = gh.calls_to("create_pull")
    assert create["base"] == "feature"
    assert create["head"].split(":")[-1] == BRANCH
    assert create["repo"].full_name == "octo/sample"
    assert create["title"].startswith(f"ci-fix: fixes for #{SAMPLE_PR}")
    assert create["body"] == result.pr_body
    assert result.pr_body.startswith(f"Fixes failing tests in #{SAMPLE_PR} (`feature`).")
    assert gh.calls_to("edit_pull") == []

    (comment,) = gh.calls_to("create_comment")
    assert comment["number"] == SAMPLE_PR
    assert COMMENT_MARKER in comment["body"]
    assert result.pr_url is not None and result.pr_url in comment["body"]
    assert result.pushed is True
    assert result.pr_url == create_url(gh)
    assert run_dirs_for(tmp_path) == []


def create_url(gh: FakeGitHub) -> str:
    (pr,) = gh.pulls.values()
    return pr["url"]


def test_custom_author(tmp_path: Path, remote: SampleRemote, gh: FakeGitHub) -> None:
    run_pipeline(
        tmp_path,
        remote,
        two_fixes(),
        [SUBTRACT],
        gh,
        commit_author_name="Bot Person",
        commit_author_email="bot@example.com",
    )
    author = git("log", "-1", "--format=%an <%ae>", BRANCH, cwd=remote.bare)
    assert author == "Bot Person <bot@example.com>"


def test_rerun_force_updates_branch_and_edits_pr_and_comment(
    tmp_path: Path, remote: SampleRemote, gh: FakeGitHub
) -> None:
    first = run_pipeline(
        tmp_path / "a", remote, ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]}), [SUBTRACT], gh
    )
    first_sha = git("rev-parse", BRANCH, cwd=remote.bare)

    second = run_pipeline(tmp_path / "b", remote, two_fixes(), [SUBTRACT, DIV_ZERO], gh)

    commits = commits_on_top(remote.bare, remote.pr_sha)
    assert len(commits) == 1, "re-run must replace, not stack on, the previous fix commit"
    assert commits[0] != first_sha
    assert DIVIDE_FIX in git("show", f"{BRANCH}:{OPS_PY}", cwd=remote.bare)

    assert len(gh.calls_to("create_pull")) == 1
    (edit,) = gh.calls_to("edit_pull")
    assert edit["body"] == second.pr_body
    assert second.pr_url == first.pr_url
    assert len(gh.pulls) == 1

    assert len(gh.calls_to("create_comment")) == 1
    assert len(gh.calls_to("edit_comment")) == 1
    assert len(gh.comments) == 1
    (only,) = gh.comments.values()
    assert COMMENT_MARKER in only["body"]


# ---- dry run / opt-outs ---------------------------------------------------------------------


def test_dry_run_commits_locally_but_writes_nothing(
    tmp_path: Path, remote: SampleRemote, gh: FakeGitHub
) -> None:
    result = run_pipeline(tmp_path, remote, two_fixes(), [SUBTRACT, DIV_ZERO], gh, push=False)

    assert result.pushed is False
    assert result.pr_url is None
    assert result.commit_sha
    assert result.commit_message.startswith(f"ci-fix: fix 2 failing tests in #{SAMPLE_PR}")
    assert result.pr_body.startswith(f"Fixes failing tests in #{SAMPLE_PR}")
    assert BRANCH not in remote_branches(remote.bare)
    assert gh.writes == []
    assert run_dirs_for(tmp_path) == []


def test_no_comment_still_opens_pr(tmp_path: Path, remote: SampleRemote, gh: FakeGitHub) -> None:
    result = run_pipeline(tmp_path, remote, two_fixes(), [SUBTRACT], gh, comment_on_pr=False)
    assert result.pushed is True and result.pr_url
    assert gh.calls_to("create_pull")
    assert gh.calls_to("create_comment") == [] and gh.calls_to("edit_comment") == []


# ---- nothing fixed --------------------------------------------------------------------------


def test_nothing_fixed_comments_only(tmp_path: Path, remote: SampleRemote, gh: FakeGitHub) -> None:
    result = run_pipeline(tmp_path, remote, no_fixes(), [SUBTRACT], gh)

    assert result.tests[0].status == OutcomeStatus.UNFIXABLE
    assert BRANCH not in remote_branches(remote.bare)
    assert gh.calls_to("create_pull") == [] and gh.calls_to("edit_pull") == []
    assert result.pr_url is None and result.pushed is False
    (comment,) = gh.calls_to("create_comment")
    assert comment["number"] == SAMPLE_PR
    assert COMMENT_MARKER in comment["body"]
    assert "needs product input" in comment["body"]
    assert run_dirs_for(tmp_path) == []


def test_nothing_fixed_rerun_edits_comment(
    tmp_path: Path, remote: SampleRemote, gh: FakeGitHub
) -> None:
    run_pipeline(tmp_path / "a", remote, no_fixes(), [SUBTRACT], gh)
    run_pipeline(tmp_path / "b", remote, no_fixes(), [SUBTRACT], gh)
    assert len(gh.calls_to("create_comment")) == 1
    assert len(gh.calls_to("edit_comment")) == 1


@pytest.mark.parametrize("overrides", [{"comment_on_pr": False}, {"push": False}])
def test_nothing_fixed_no_comment_when_disabled(
    tmp_path: Path, remote: SampleRemote, gh: FakeGitHub, overrides: dict[str, bool]
) -> None:
    run_pipeline(tmp_path, remote, no_fixes(), [SUBTRACT], gh, **overrides)
    assert gh.writes == []
    assert BRANCH not in remote_branches(remote.bare)


# ---- fork PRs -------------------------------------------------------------------------------


def fork_info(remote: SampleRemote, fork_url: str | None, can_modify: bool) -> PullRequestInfo:
    return sample_pr_info(remote).model_copy(
        update={
            "is_fork": True,
            "head_repo_full_name": "contrib/sample",
            "head_clone_url": fork_url,
            "maintainer_can_modify": can_modify,
        }
    )


@pytest.mark.parametrize(("fork_url", "can_modify"), [("FORK", False), (None, True), (None, False)])
def test_fork_without_permission_does_not_push(
    tmp_path: Path, remote: SampleRemote, fork_url: str | None, can_modify: bool
) -> None:
    fork = tmp_path / "fork.git"
    git("init", "-q", "--bare", str(fork))
    url = str(fork) if fork_url == "FORK" else None
    gh = FakeGitHub(fork_info(remote, url, can_modify))

    result = run_pipeline(tmp_path, remote, two_fixes(), [SUBTRACT], gh)

    assert result.tests[0].status == OutcomeStatus.FIXED
    assert result.pushed is False
    assert result.pr_url is None
    assert any(w.startswith("fork PR: cannot push") for w in result.warnings), result.warnings
    assert remote_branches(fork) == []
    assert BRANCH not in remote_branches(remote.bare)
    assert gh.calls_to("create_pull") == []
    assert result.diff  # the fix is still returned as a diff
    assert run_dirs_for(tmp_path) == []


def test_fork_with_permission_pushes_to_fork(tmp_path: Path, remote: SampleRemote) -> None:
    fork = tmp_path / "fork.git"
    git("init", "-q", "--bare", str(fork))
    gh = FakeGitHub(fork_info(remote, str(fork), True))

    result = run_pipeline(tmp_path, remote, two_fixes(), [SUBTRACT], gh)

    assert result.pushed is True
    assert BRANCH in remote_branches(fork)
    assert BRANCH not in remote_branches(remote.bare)
    assert git("rev-parse", f"{BRANCH}^", cwd=fork) == remote.pr_sha
    (create,) = gh.calls_to("create_pull")
    assert create["repo"].full_name == "contrib/sample"
    assert create["base"] == "feature"
    assert result.pr_url == create_url(gh)


# ---- failures -------------------------------------------------------------------------------


def _make_readonly(path: Path) -> list[Path]:
    changed = []
    for root, dirs, files in os.walk(path):
        for name in [*dirs, *files, "."]:
            p = Path(root) / name
            mode = p.stat().st_mode
            p.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
            changed.append(p)
    return changed


def _restore_writable(path: Path) -> None:
    subprocess.run(["chmod", "-R", "u+w", str(path)], check=False)


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_push_failure_raises_git_error_and_cleans_up(
    tmp_path: Path, remote: SampleRemote, gh: FakeGitHub
) -> None:
    _make_readonly(remote.bare)
    try:
        with pytest.raises(GitError) as exc:
            run_pipeline(tmp_path, remote, two_fixes(), [SUBTRACT], gh)
    finally:
        _restore_writable(remote.bare)
    message = str(exc.value)
    assert "push" in message.lower()
    assert BRANCH in message
    assert gh.calls_to("create_pull") == []
    assert run_dirs_for(tmp_path) == []


def test_push_to_missing_fork_raises_git_error_and_cleans_up(
    tmp_path: Path, remote: SampleRemote
) -> None:
    gh = FakeGitHub(fork_info(remote, str(tmp_path / "does-not-exist.git"), True))
    with pytest.raises(GitError) as exc:
        run_pipeline(tmp_path, remote, two_fixes(), [SUBTRACT], gh)
    assert "push" in str(exc.value).lower()
    assert "contrib/sample" in str(exc.value)
    assert gh.calls_to("create_pull") == []
    assert run_dirs_for(tmp_path) == []


def test_push_permission_error_has_token_hint(tmp_path: Path) -> None:
    """GitRepo.push adds a token-scope hint when the remote refuses with 403."""
    from ci_fix.tools import git as git_tools

    repo = git_tools.GitRepo(tmp_path)

    def refuse(*args: Any, **kw: Any) -> str:
        raise GitError("git command failed (exit 128): remote: Permission denied (403)")

    repo._git = refuse  # type: ignore[method-assign]
    with pytest.raises(GitError) as exc:
        repo.push("origin", BRANCH, BRANCH, token=None, label="octo/sample")
    assert "hint" in str(exc.value) and "octo/sample" in str(exc.value)


class SampleClient(GitHubClient):
    """The real client (over a mocked PyGithub) with a canned ``get_pull_request``."""

    def __init__(self, info: PullRequestInfo, pygithub: MagicMock) -> None:
        super().__init__(github=pygithub)
        self.info = info

    def get_pull_request(self, repo: RepoRef, number: int) -> PullRequestInfo:
        return self.info


@pytest.mark.parametrize("status", [403, 422])
def test_pr_creation_error_raises_github_error_with_hint(
    tmp_path: Path, remote: SampleRemote, status: int
) -> None:
    pygithub = MagicMock()
    repo_api = pygithub.get_repo.return_value
    repo_api.get_pulls.return_value = []
    repo_api.create_pull.side_effect = GithubException(status, {"message": "nope"}, None)
    client = SampleClient(sample_pr_info(remote), pygithub)

    with pytest.raises(GitHubError) as exc:
        run_pipeline(tmp_path, remote, two_fixes(), [SUBTRACT], client)
    message = str(exc.value)
    assert str(status) in message
    assert "hint" in message and "Pull requests" in message
    assert run_dirs_for(tmp_path) == []
    # The branch was pushed before the PR call failed.
    assert BRANCH in remote_branches(remote.bare)


def test_real_client_delivery_calls(tmp_path: Path, remote: SampleRemote) -> None:
    """The real GitHubClient: get_pulls → create_pull → issue comment, with the right args."""
    pygithub = MagicMock()
    repo_api = pygithub.get_repo.return_value
    repo_api.get_pulls.return_value = []
    repo_api.create_pull.return_value = MagicMock(
        number=77, html_url="https://github.com/octo/sample/pull/77"
    )
    issue = repo_api.get_issue.return_value
    issue.get_comments.return_value = [MagicMock(body="unrelated", id=1)]
    client = SampleClient(sample_pr_info(remote), pygithub)

    result = run_pipeline(tmp_path, remote, two_fixes(), [SUBTRACT], client)

    assert result.pr_url == "https://github.com/octo/sample/pull/77"
    kwargs = repo_api.create_pull.call_args.kwargs
    assert kwargs["base"] == "feature"
    assert kwargs["head"].split(":")[-1] == BRANCH
    pygithub.get_repo.assert_any_call("octo/sample")
    repo_api.get_issue.assert_any_call(SAMPLE_PR)
    (body,) = issue.create_comment.call_args.args
    assert COMMENT_MARKER in body


def test_real_client_edits_existing_pr_and_comment(tmp_path: Path, remote: SampleRemote) -> None:
    pygithub = MagicMock()
    repo_api = pygithub.get_repo.return_value
    pygithub.get_user.return_value.login = "ci-bot"
    existing_pr = MagicMock(
        number=77,
        html_url="https://github.com/octo/sample/pull/77",
        body="old\n---\nGenerated by ci-fix for #11.",
    )
    repo_api.get_pulls.return_value = [existing_pr]
    repo_api.get_pull.return_value = existing_pr
    issue = repo_api.get_issue.return_value
    comment = MagicMock(body=f"{COMMENT_MARKER}\nold", id=9)
    comment.user.login = "ci-bot"
    issue.get_comments.return_value = [comment]
    client = SampleClient(sample_pr_info(remote), pygithub)

    result = run_pipeline(tmp_path, remote, two_fixes(), [SUBTRACT], client)

    assert result.pr_url == "https://github.com/octo/sample/pull/77"
    repo_api.create_pull.assert_not_called()
    existing_pr.edit.assert_called_once()
    issue.create_comment.assert_not_called()
    issue.get_comment.assert_called_once_with(9)
    issue.get_comment.return_value.edit.assert_called_once()


def test_comment_failure_is_a_warning_not_an_error(
    tmp_path: Path, remote: SampleRemote, gh: FakeGitHub
) -> None:
    gh.fail["create_comment"] = GitHubError("HTTP 403 nope")
    result = run_pipeline(tmp_path, remote, two_fixes(), [SUBTRACT], gh)
    assert result.pr_url and result.pushed
    assert any("comment" in w.lower() for w in result.warnings)


# ---- secrets --------------------------------------------------------------------------------


def test_token_never_logged(
    tmp_path: Path, remote: SampleRemote, gh: FakeGitHub, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="ci_fix")
    result = run_pipeline(
        tmp_path, remote, two_fixes(), [SUBTRACT, DIV_ZERO], gh, github_token=FAKE_TOKEN
    )
    assert result.pushed is True
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "[deliver]" in text or "push" in text.lower()
    assert FAKE_TOKEN not in text
    assert FAKE_TOKEN not in caplog.text
    dumped = result.model_dump_json()
    assert FAKE_TOKEN not in dumped
    for _, kw in gh.calls:
        assert FAKE_TOKEN not in repr(kw)
    # Nor persisted in the pushed commit or the remote's config.
    assert FAKE_TOKEN not in git("log", "--format=%B", BRANCH, cwd=remote.bare)
    assert FAKE_TOKEN not in (remote.bare / "config").read_text()


def test_token_not_logged_on_push_failure(
    tmp_path: Path, remote: SampleRemote, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="ci_fix")
    gh = FakeGitHub(fork_info(remote, str(tmp_path / "missing.git"), True))
    with pytest.raises(GitError) as exc:
        run_pipeline(tmp_path, remote, two_fixes(), [SUBTRACT], gh, github_token=FAKE_TOKEN)
    assert FAKE_TOKEN not in str(exc.value)
    assert FAKE_TOKEN not in caplog.text


# ---- guards (review fixes) ------------------------------------------------------------------


@pytest.mark.parametrize("field_name", ["head_ref", "base_ref", "default"])
def test_refuses_to_push_to_pr_or_default_branch(
    tmp_path: Path, remote: SampleRemote, field_name: str
) -> None:
    info = sample_pr_info(remote)
    gh = FakeGitHub(info)
    if field_name == "default":
        gh.default_branch = BRANCH
    else:
        gh.info = info.model_copy(update={field_name: BRANCH})
    with pytest.raises(GitHubError, match="refusing to push"):
        run_pipeline(tmp_path, remote, two_fixes(), [SUBTRACT], gh)
    assert BRANCH not in remote_branches(remote.bare)
    assert gh.writes == []


def test_existing_foreign_branch_is_not_overwritten(
    tmp_path: Path, remote: SampleRemote, gh: FakeGitHub
) -> None:
    git("update-ref", f"refs/heads/{BRANCH}", remote.pr_sha, cwd=remote.bare)  # by "Tester"
    with pytest.raises(GitHubError, match="was not created by ci-fix"):
        run_pipeline(tmp_path, remote, two_fixes(), [SUBTRACT], gh)
    assert git("rev-parse", BRANCH, cwd=remote.bare) == remote.pr_sha
    assert gh.writes == []
    assert run_dirs_for(tmp_path) == []


def test_existing_branch_by_other_author_email_is_not_overwritten(
    tmp_path: Path, remote: SampleRemote, gh: FakeGitHub
) -> None:
    run_pipeline(tmp_path / "a", remote, two_fixes(), [SUBTRACT], gh)
    tip = git("rev-parse", BRANCH, cwd=remote.bare)
    gh2 = FakeGitHub(sample_pr_info(remote))
    with pytest.raises(GitHubError, match="was not created by ci-fix"):
        run_pipeline(
            tmp_path / "b", remote, two_fixes(), [SUBTRACT], gh2, commit_author_email="x@y.z"
        )
    assert git("rev-parse", BRANCH, cwd=remote.bare) == tip


def test_head_moved_still_pushes_with_warning_and_body_line(
    tmp_path: Path, remote: SampleRemote, gh: FakeGitHub
) -> None:
    gh.moved_head = "f" * 40
    result = run_pipeline(tmp_path, remote, two_fixes(), [SUBTRACT], gh)
    assert result.pushed and result.pr_url
    assert any("head moved" in w for w in result.warnings), result.warnings
    (created,) = gh.calls_to("create_pull")
    first = created["body"].splitlines()[0]
    assert first == (
        f"⚠ verified against `{remote.pr_sha[:12]}`; `feature` has since moved to "
        f"`{'f' * 12}` — re-run ci-fix"
    )
    assert result.pr_body == created["body"] == result.summary


def test_head_not_moved_no_warning(tmp_path: Path, remote: SampleRemote, gh: FakeGitHub) -> None:
    result = run_pipeline(tmp_path, remote, two_fixes(), [SUBTRACT], gh)
    assert not any("head moved" in w for w in result.warnings)
    assert gh.calls_to("create_pull")[0]["body"].startswith("Fixes failing tests in #")


def test_llm_text_is_sanitized_in_delivered_texts(
    tmp_path: Path, remote: SampleRemote, gh: FakeGitHub
) -> None:
    explanation = "Root cause: @octocat broke it <img src=x>\nFix: fixes #1 properly"
    fixer = ScriptedFixer({SUBTRACT: [(*FIX_SUBTRACT[:4], explanation)]})
    result = run_pipeline(tmp_path, remote, fixer, [SUBTRACT], gh)
    body = gh.calls_to("create_pull")[0]["body"]
    for text in (body, result.commit_message):
        assert "@​octocat" in text and "@octocat" not in text
        assert "<img" not in text
        assert "fixes ​#1" in text
    (comment,) = gh.calls_to("create_comment")
    assert comment["body"].startswith(COMMENT_MARKER)


# ---- real client: ownership of PRs and comments --------------------------------------------


def _pygithub(login: str | None = "ci-bot") -> MagicMock:
    pygithub = MagicMock()
    if login is None:
        pygithub.get_user.side_effect = GithubException(403, {"message": "app token"}, None)
    else:
        pygithub.get_user.return_value.login = login
    return pygithub


def _comment(body: str, author: str, cid: int) -> MagicMock:
    c = MagicMock(body=body, id=cid)
    c.user.login = author
    return c


REF = RepoRef(owner="octo", name="sample")


def test_find_comment_needs_marker_at_start_and_own_author() -> None:
    pygithub = _pygithub("ci-bot")
    pygithub.get_repo.return_value.get_issue.return_value.get_comments.return_value = [
        _comment(f"quoting {COMMENT_MARKER}", "ci-bot", 1),  # marker not at start
        _comment(f"{COMMENT_MARKER}\nspoofed", "mallory", 2),  # someone else
        _comment(f"{COMMENT_MARKER}\nours", "ci-bot", 3),
    ]
    assert GitHubClient(github=pygithub).find_comment(REF, 11, COMMENT_MARKER) == 3


def test_find_comment_none_when_only_foreign() -> None:
    pygithub = _pygithub("ci-bot")
    pygithub.get_repo.return_value.get_issue.return_value.get_comments.return_value = [
        _comment(f"{COMMENT_MARKER}\nspoofed", "mallory", 2)
    ]
    assert GitHubClient(github=pygithub).find_comment(REF, 11, COMMENT_MARKER) is None


def test_find_comment_app_token_falls_back_to_marker(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger="ci_fix")
    pygithub = _pygithub(None)
    pygithub.get_repo.return_value.get_issue.return_value.get_comments.return_value = [
        _comment(f"x {COMMENT_MARKER}", "bot[bot]", 1),
        _comment(f"{COMMENT_MARKER}\nours", "bot[bot]", 2),
    ]
    client = GitHubClient(github=pygithub)
    assert client.find_comment(REF, 11, COMMENT_MARKER) == 2
    assert client.find_comment(REF, 11, COMMENT_MARKER) == 2
    assert pygithub.get_user.call_count == 1  # cached
    assert "authenticated GitHub user" in caplog.text


def _pr(body: str, author: str) -> MagicMock:
    pr = MagicMock(number=7, html_url="https://github.com/octo/sample/pull/7", body=body)
    pr.user.login = author
    return pr


@pytest.mark.parametrize(
    ("body", "author"),
    [("x\n---\nGenerated by ci-fix for #11 (u).", "someone"), ("edited by hand", "ci-bot")],
)
def test_find_open_pr_accepts_ci_fix_prs(body: str, author: str) -> None:
    pygithub = _pygithub("ci-bot")
    pygithub.get_repo.return_value.get_pulls.return_value = [_pr(body, author)]
    ref = GitHubClient(github=pygithub).find_open_pr(REF, f"octo:{BRANCH}", "feature")
    assert ref == PullRef(number=7, html_url="https://github.com/octo/sample/pull/7")


def test_find_open_pr_foreign_pr_raises() -> None:
    pygithub = _pygithub("ci-bot")
    pygithub.get_repo.return_value.get_pulls.return_value = [_pr("hand-made", "mallory")]
    with pytest.raises(GitHubError, match="already exists and was not created by ci-fix"):
        GitHubClient(github=pygithub).find_open_pr(REF, f"octo:{BRANCH}", "feature")


def test_api_error_data_truncated() -> None:
    pygithub = _pygithub("ci-bot")
    pygithub.get_repo.return_value.create_pull.side_effect = GithubException(
        422, {"message": "x" * 5000}, None
    )
    with pytest.raises(GitHubError) as exc:
        GitHubClient(github=pygithub).create_pull(REF, "t", "b", BRANCH, "feature")
    assert len(str(exc.value)) < 800
    assert "Pull requests: write" in str(exc.value)
