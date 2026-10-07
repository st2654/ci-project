"""Tests for the PR context given to the fixer (slice 4).

Covers ``PullRequestInfo.body``, ``GitRepo.fetch_base``/``merge_base``,
``PreparedRepo.base_sha`` and the new ``FixRequest`` fields the graph fills in
(PR title/body/diff, other failing tests and the ``run_test`` callback).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from conftest import REPO_URL as FAKE_REPO_URL
from conftest import FakeRemote, fake_client, git, pr_info
from fixtures import copy_sample_repo
from pipeline_helpers import (
    ADD,
    FIX_SUBTRACT,
    MEAN,
    OPS_PY,
    REPO_URL,
    SAMPLE_PR,
    SUBTRACT,
    FakeRunnerFactory,
    SampleRemote,
    ScriptedFixer,
    make_deps,
    sample_pr_info,
)

from ci_fix.config import Settings
from ci_fix.models import FixAttempt, FixRequest, OutcomeStatus
from ci_fix.pipeline import fix_failing_tests
from ci_fix.tools.git import GitError, GitRepo
from ci_fix.tools.github import GitHubClient, PullRequestInfo, RepoRef
from ci_fix.tools.pytest_runner import TestResult, TestStatus
from ci_fix.workspace import prepare_pr_checkout

TRUNCATED_RE = re.compile(r"\[\.\.\. truncated (\d+) chars\]")


# --------------------------------------------------------------------------- #
# PullRequestInfo.body
# --------------------------------------------------------------------------- #


def test_pull_request_info_body_defaults_to_empty() -> None:
    info = PullRequestInfo(
        number=1,
        title="t",
        state="open",
        base_ref="main",
        head_ref="f",
        head_sha="a" * 40,
        head_repo_full_name="o/r",
        is_fork=False,
        html_url="https://github.com/o/r/pull/1",
    )
    assert info.body == ""


def _github_pr(body: Any) -> MagicMock:
    pr = MagicMock()
    pr.number = 7
    pr.title = "Fix the thing"
    pr.body = body
    pr.state = "open"
    pr.base.ref = "main"
    pr.base.repo.full_name = "octo/repo"
    pr.head.ref = "feature"
    pr.head.sha = "a" * 40
    pr.head.repo.full_name = "octo/repo"
    pr.head.repo.clone_url = "https://github.com/octo/repo.git"
    pr.maintainer_can_modify = False
    pr.html_url = "https://github.com/octo/repo/pull/7"
    return pr


@pytest.mark.parametrize(
    ("body", "expected"), [("Some **markdown**", "Some **markdown**"), (None, "")]
)
def test_github_client_maps_body(body: str | None, expected: str) -> None:
    gh = MagicMock()
    gh.get_repo.return_value.get_pull.return_value = _github_pr(body)
    info = GitHubClient(github=gh).get_pull_request(RepoRef(owner="octo", name="repo"), 7)
    assert info.body == expected


# --------------------------------------------------------------------------- #
# GitRepo.fetch_base / merge_base
# --------------------------------------------------------------------------- #


@pytest.fixture
def cloned(fake_remote: FakeRemote, tmp_path: Path) -> GitRepo:
    repo = GitRepo.clone(fake_remote.url, tmp_path / "clone")
    repo.fetch_pr(fake_remote.pr_number)
    return repo


def test_fetch_base_returns_sha_and_sets_ref(cloned: GitRepo, fake_remote: FakeRemote) -> None:
    sha = cloned.fetch_base("main")
    assert sha == fake_remote.main_sha
    assert git("rev-parse", "refs/ci-fix-fetch/base", cwd=cloned.path) == fake_remote.main_sha


def test_fetch_base_missing_branch_raises(cloned: GitRepo) -> None:
    with pytest.raises(GitError):
        cloned.fetch_base("no-such-branch-xyz")


def test_merge_base_of_pr_and_base(cloned: GitRepo, fake_remote: FakeRemote) -> None:
    base = cloned.fetch_base("main")
    assert cloned.merge_base(base, fake_remote.pr_sha) == fake_remote.main_sha
    assert cloned.merge_base(fake_remote.pr_sha, base) == fake_remote.main_sha


# --------------------------------------------------------------------------- #
# A sample remote with a configurable PR change (and optionally a moved base)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Remote:
    sample: SampleRemote
    fork_point: str
    main_sha: str


def build_remote(
    tmp_path: Path,
    pr_files: dict[str, str],
    main_after: dict[str, str] | None = None,
) -> Remote:
    """Sample repo on ``main``; the PR adds ``pr_files``; ``main_after`` lands on main later."""
    work = copy_sample_repo(tmp_path / "origin-work")
    git("init", "-q", "-b", "main", cwd=work)
    git("config", "user.email", "tester@example.com", cwd=work)
    git("config", "user.name", "Tester", cwd=work)
    git("config", "commit.gpgsign", "false", cwd=work)
    git("add", "-A", cwd=work)
    git("commit", "-q", "-m", "initial", cwd=work)
    fork_point = git("rev-parse", "HEAD", cwd=work)

    git("checkout", "-q", "-b", "feature", cwd=work)
    for rel, text in pr_files.items():
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_text(text, encoding="utf-8")
    git("add", "-A", cwd=work)
    git("commit", "-q", "-m", "pr change", cwd=work)
    pr_sha = git("rev-parse", "HEAD", cwd=work)
    git("checkout", "-q", "main", cwd=work)

    if main_after:
        for rel, text in main_after.items():
            (work / rel).write_text(text, encoding="utf-8")
        git("add", "-A", cwd=work)
        git("commit", "-q", "-m", "later change on main", cwd=work)
    main_sha = git("rev-parse", "HEAD", cwd=work)

    bare = tmp_path / "remote.git"
    git("clone", "-q", "--bare", str(work), str(bare))
    git("update-ref", f"refs/pull/{SAMPLE_PR}/head", pr_sha, cwd=bare)
    return Remote(SampleRemote(bare=bare, pr_number=SAMPLE_PR, pr_sha=pr_sha), fork_point, main_sha)


def _github(remote: SampleRemote, **updates: Any) -> MagicMock:
    info = sample_pr_info(remote).model_copy(update=updates)
    github = MagicMock()
    github.get_pull_request.return_value = info
    return github


# --------------------------------------------------------------------------- #
# PreparedRepo.base_sha
# --------------------------------------------------------------------------- #


def test_prepared_base_sha_is_merge_base(fake_remote: FakeRemote, tmp_path: Path) -> None:
    prepared = prepare_pr_checkout(
        FAKE_REPO_URL,
        fake_remote.pr_number,
        Settings(workspace_dir=tmp_path / "ws"),
        fake_client(pr_info(fake_remote)),
        clone_url=fake_remote.url,
    )
    assert prepared.base_sha == fake_remote.main_sha


def test_prepared_base_sha_is_fork_point_when_base_moved(tmp_path: Path) -> None:
    remote = build_remote(
        tmp_path, {"PR_CHANGE.txt": "pr\n"}, main_after={"README.md": "main moved on\n"}
    )
    assert remote.main_sha != remote.fork_point
    prepared = prepare_pr_checkout(
        REPO_URL,
        SAMPLE_PR,
        Settings(workspace_dir=tmp_path / "ws"),
        _github(remote.sample),
        clone_url=remote.sample.url,
    )
    assert prepared.base_sha == remote.fork_point


def test_base_fetch_failure_gives_none_and_warning(
    fake_remote: FakeRemote, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="ci_fix")
    info = pr_info(fake_remote).model_copy(update={"base_ref": "no-such-base"})
    prepared = prepare_pr_checkout(
        FAKE_REPO_URL,
        fake_remote.pr_number,
        Settings(workspace_dir=tmp_path / "ws"),
        fake_client(info),
        clone_url=fake_remote.url,
    )
    assert prepared.base_sha is None
    assert prepared.pr_head_sha == fake_remote.pr_sha  # the run goes on
    assert any(r.levelno == logging.WARNING for r in caplog.records)


# --------------------------------------------------------------------------- #
# FixRequest model
# --------------------------------------------------------------------------- #


def test_fix_request_run_test_not_serialized(tmp_path: Path) -> None:
    def run_test(node_id: str) -> TestResult:
        return TestResult(node_id=node_id, status=TestStatus.PASSED)

    request = FixRequest(
        node_id=SUBTRACT,
        repo_path=tmp_path,
        attempt=1,
        max_attempts=3,
        failure_message="m",
        failure_details="d",
        pr_title="t",
        pr_body="b",
        pr_diff="diff",
        other_failing_tests=[MEAN],
        run_test=run_test,
    )
    assert request.run_test is run_test
    dumped = request.model_dump()
    assert "run_test" not in dumped
    assert dumped["pr_title"] == "t"
    assert dumped["other_failing_tests"] == [MEAN]
    assert "run_test" not in json.loads(request.model_dump_json())


# --------------------------------------------------------------------------- #
# Through the pipeline
# --------------------------------------------------------------------------- #


@dataclass
class CapturingFixer:
    """Wraps ``ScriptedFixer``: records each request and calls ``run_test`` as told."""

    inner: ScriptedFixer
    run_ids: dict[str, list[str]] = field(default_factory=dict)  # target -> ids to run first
    run_after: dict[str, list[str]] = field(default_factory=dict)  # target -> ids after edits
    requests: list[FixRequest] = field(default_factory=list)
    results: list[tuple[str, str, TestResult]] = field(default_factory=list)

    def _run(self, request: FixRequest, ids: list[str]) -> None:
        assert request.run_test is not None
        for nid in ids:
            self.results.append((request.node_id, nid, request.run_test(nid)))

    def fix(self, request: FixRequest) -> FixAttempt:
        self.requests.append(request)
        self._run(request, self.run_ids.get(request.node_id, []))
        attempt = self.inner.fix(request)
        self._run(request, self.run_after.get(request.node_id, []))
        return attempt


def _run(tmp_path: Path, remote: SampleRemote, fixer: Any, tests: list[str], **kw: Any):
    github = kw.pop("github", None)
    factory = kw.pop("runner_factory", None) or FakeRunnerFactory()
    deps = make_deps(tmp_path, fixer, remote=remote, runner_factory=factory, **kw)
    if github is not None:
        deps = dataclasses.replace(deps, github=github)
    return fix_failing_tests(REPO_URL, SAMPLE_PR, tests, deps=deps), factory


def test_request_has_pr_title_body_and_diff(tmp_path: Path, sample_remote: SampleRemote) -> None:
    fixer = CapturingFixer(ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]}))
    github = _github(sample_remote, title="Add PR_CHANGE", body="Why: we need the change.")
    result, _ = _run(tmp_path, sample_remote, fixer, [SUBTRACT], github=github)

    assert result.tests[0].status == OutcomeStatus.FIXED
    (req,) = fixer.requests
    assert req.pr_title == "Add PR_CHANGE"
    assert req.pr_body == "Why: we need the change."
    assert "PR_CHANGE.txt" in req.pr_diff
    assert "+change from the PR" in req.pr_diff
    assert "ops.py" not in req.pr_diff  # the PR did not touch it
    assert not TRUNCATED_RE.search(req.pr_diff)


def test_pr_diff_excludes_later_base_changes(tmp_path: Path) -> None:
    remote = build_remote(
        tmp_path,
        {"PR_CHANGE.txt": "PR-ONLY-LINE\n"},
        main_after={"README.md": "MAIN-ONLY-LINE\n"},
    )
    fixer = CapturingFixer(ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]}))
    _run(tmp_path, remote.sample, fixer, [SUBTRACT])
    (req,) = fixer.requests
    assert "PR-ONLY-LINE" in req.pr_diff
    assert "MAIN-ONLY-LINE" not in req.pr_diff
    assert "README.md" not in req.pr_diff


def test_pr_diff_truncated(tmp_path: Path) -> None:
    big = "".join(f"generated line {i:05d} with some padding text\n" for i in range(3000))
    remote = build_remote(tmp_path, {"data/big.txt": big})
    fixer = CapturingFixer(ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]}))
    result, _ = _run(tmp_path, remote.sample, fixer, [SUBTRACT], pr_diff_max_chars=1000)

    assert result.tests[0].status == OutcomeStatus.FIXED
    (req,) = fixer.requests
    match = TRUNCATED_RE.search(req.pr_diff)
    assert match, req.pr_diff[-200:]
    kept = req.pr_diff[: match.start()].rstrip("\n")
    assert len(kept) <= 1000
    assert len(req.pr_diff) <= 1000 + len(match.group(0)) + 1
    assert "data/big.txt" in kept
    assert int(match.group(1)) > len(big) // 2  # most of the diff was dropped


def test_pr_diff_empty_when_base_fetch_fails(
    tmp_path: Path, sample_remote: SampleRemote, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="ci_fix")
    fixer = CapturingFixer(ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]}))
    github = _github(sample_remote, base_ref="no-such-base")
    result, _ = _run(tmp_path, sample_remote, fixer, [SUBTRACT], github=github)

    assert result.tests[0].status == OutcomeStatus.FIXED  # the run continues
    (req,) = fixer.requests
    assert isinstance(req.pr_diff, str)
    assert "+change from the PR" not in req.pr_diff
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_other_failing_tests(tmp_path: Path, sample_remote: SampleRemote) -> None:
    fixer = CapturingFixer(ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]}))
    _run(tmp_path, sample_remote, fixer, [SUBTRACT, MEAN, ADD])
    first = fixer.requests[0]
    assert first.node_id == SUBTRACT
    assert MEAN in first.other_failing_tests
    assert SUBTRACT not in first.other_failing_tests
    assert ADD not in first.other_failing_tests  # passes already
    mean_requests = [r for r in fixer.requests if r.node_id == MEAN]
    assert mean_requests
    assert SUBTRACT not in mean_requests[0].other_failing_tests  # fixed by then
    assert MEAN not in mean_requests[0].other_failing_tests


def test_run_test_callback_runs_target(tmp_path: Path, sample_remote: SampleRemote) -> None:
    fixer = CapturingFixer(
        ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]}),
        run_ids={SUBTRACT: [SUBTRACT]},
        run_after={SUBTRACT: [SUBTRACT]},
    )
    result, factory = _run(tmp_path, sample_remote, fixer, [SUBTRACT])
    before, after = (r for _, _, r in fixer.results)
    assert before.status == TestStatus.FAILED
    assert after.status == TestStatus.PASSED
    assert result.tests[0].status == OutcomeStatus.FIXED
    assert [SUBTRACT] in factory.runners[0].runs


def test_run_test_callback_refuses_other_ids(tmp_path: Path, sample_remote: SampleRemote) -> None:
    fixer = CapturingFixer(
        ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]}), run_ids={SUBTRACT: [MEAN, ADD]}
    )
    _, factory = _run(tmp_path, sample_remote, fixer, [SUBTRACT, MEAN])
    refused = [r for target, nid, r in fixer.results if target == SUBTRACT]
    assert [r.status for r in refused] == [TestStatus.NOT_FOUND, TestStatus.NOT_FOUND]
    assert all("only the target test can be run" in r.message for r in refused)
    # The refused ids were never sent to the runner on their own.
    assert [ADD] not in factory.runners[0].runs


def test_run_test_artifacts_are_not_fixer_changes(
    tmp_path: Path, sample_remote: SampleRemote
) -> None:
    fixer = CapturingFixer(
        ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]}),
        run_ids={SUBTRACT: [SUBTRACT]},
        run_after={SUBTRACT: [SUBTRACT, SUBTRACT]},
    )
    factory = FakeRunnerFactory(artifacts=True)
    result, _ = _run(tmp_path, sample_remote, fixer, [SUBTRACT], runner_factory=factory)

    (outcome,) = result.tests
    assert outcome.status == OutcomeStatus.FIXED
    assert outcome.files_changed == [OPS_PY]
    assert "out/run-" not in result.diff
    assert ".coverage" not in result.diff


def test_run_test_only_attempt_is_no_change(tmp_path: Path, sample_remote: SampleRemote) -> None:
    """A fixer that only runs the test (leaving artifacts) made no change."""
    fixer = CapturingFixer(ScriptedFixer({}), run_ids={SUBTRACT: [SUBTRACT, SUBTRACT]})
    factory = FakeRunnerFactory(artifacts=True)
    result, _ = _run(
        tmp_path, sample_remote, fixer, [SUBTRACT], runner_factory=factory, max_attempts=1
    )
    (outcome,) = result.tests
    assert outcome.status == OutcomeStatus.UNFIXABLE
    assert outcome.files_changed == []
    assert result.diff == ""
