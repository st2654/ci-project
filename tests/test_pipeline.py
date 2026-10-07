"""Graph-logic tests for the pipeline (fix_failing_tests + the LangGraph loop).

Fast: the remote is a session-wide local bare repo, GitHub is a mock, and tests run on a
``FakeRunner`` whose statuses are derived from the checkout's current file contents.
End-to-end tests on the real pytest live in ``test_pipeline_e2e.py``.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from pathlib import Path

import pytest
from conftest import git
from pipeline_helpers import (
    ADD,
    ADD_BODY,
    DELETE_SUBTRACT_TEST,
    DIV_OK,
    DIV_ZERO,
    DIVIDE_BUG,
    DIVIDE_FIX,
    FIX_DIVIDE,
    FIX_MEAN,
    FIX_SUBTRACT,
    FIXTURE_ERROR,
    MEAN,
    MEAN_BUG,
    MEAN_FIX,
    OPS_PY,
    OTHER_PR,
    REPO_URL,
    SAMPLE_PR,
    SKIPPED,
    SUBTRACT,
    SUBTRACT_BUG,
    SUBTRACT_FIX,
    SUBTRACT_TEST,
    TEST_OPS_PY,
    WRONG_SUBTRACT,
    FakeRunnerFactory,
    SampleRemote,
    ScriptedFixer,
    StatelessFixer,
    fake_env_factory,
    flaky,
    make_deps,
    marker_rule,
    run_dir_for,
    run_dirs_for,
)

from ci_fix.models import FixAttempt, FixRequest, FixResult, NoOpFixer, OutcomeStatus, TestOutcome
from ci_fix.pipeline import fix_failing_tests
from ci_fix.tools.git import GitError
from ci_fix.tools.pytest_runner import TestStatus
from ci_fix.tools.test_env import TestEnvError

BREAK_ADD_AND_FIX_SUBTRACT = (
    "edits",
    [
        (OPS_PY, ADD_BODY, ADD_BODY.replace("a + b\n", "a - b\n")),
        (OPS_PY, SUBTRACT_BUG, SUBTRACT_FIX),
    ],
    "subtract everywhere",
)
FIX_SUBTRACT_AND_DIVIDE = (
    "edits",
    [(OPS_PY, SUBTRACT_BUG, SUBTRACT_FIX), (OPS_PY, DIVIDE_BUG, DIVIDE_FIX)],
    "fix both",
)


@pytest.fixture
def runners() -> FakeRunnerFactory:
    return FakeRunnerFactory()


@pytest.fixture
def run(tmp_path: Path, sample_remote: SampleRemote, runners: FakeRunnerFactory):
    def _run(fixer, tests: list[str], **overrides) -> FixResult:
        deps = make_deps(tmp_path, fixer, remote=sample_remote, runner_factory=runners, **overrides)
        return fix_failing_tests(REPO_URL, SAMPLE_PR, tests, deps=deps)

    return _run


def _only(result: FixResult) -> TestOutcome:
    assert len(result.tests) == 1
    return result.tests[0]


def _by_name(result: FixResult) -> dict[str, TestOutcome]:
    return {t.requested_name: t for t in result.tests}


def _checkpoints(tmp_path: Path, remote: SampleRemote) -> list[str]:
    repo = run_dir_for(tmp_path) / "repo"
    log = git(
        "log", "--format=%s", "--reverse", f"{remote.pr_sha}..refs/ci-fix/checkpoints", cwd=repo
    )
    return log.splitlines()


# ---- accept / reject ------------------------------------------------------------------------


def test_fixed_on_first_attempt(run, tmp_path: Path) -> None:
    fixer = ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]})
    result = run(fixer, [SUBTRACT])

    outcome = _only(result)
    assert outcome.status == OutcomeStatus.FIXED
    assert (outcome.node_id, outcome.attempts, outcome.files_changed) == (SUBTRACT, 1, [OPS_PY])
    assert result.fixed == [outcome]
    assert result.branch == f"ci-fix/pr-{SAMPLE_PR}"
    assert result.pr_url is None
    assert "+    return a - b" in result.diff
    assert "PR_CHANGE.txt" not in result.diff  # diff is against the PR head, not main
    assert SUBTRACT in result.summary

    (req,) = fixer.requests
    assert (req.node_id, req.attempt, req.max_attempts, req.previous_attempts) == (
        SUBTRACT,
        1,
        3,
        [],
    )
    assert SUBTRACT in req.failure_message
    run_dir = Path(req.repo_path).parent
    assert run_dir.parent == tmp_path / "ws"
    assert run_dir.name.startswith(f"octo__sample__pr-{SAMPLE_PR}__")
    assert Path(req.repo_path).name == "repo"


def test_rejected_attempt_is_rolled_back_and_reason_fed_to_next(run) -> None:
    # The 2nd action edits SUBTRACT_BUG again: it only applies if attempt 1 was rolled back.
    fixer = ScriptedFixer({SUBTRACT: [WRONG_SUBTRACT, FIX_SUBTRACT]})
    result = run(fixer, [SUBTRACT])

    outcome = _only(result)
    assert (outcome.status, outcome.attempts) == (OutcomeStatus.FIXED, 2)
    assert "a * b" not in result.diff

    first, second = fixer.requests
    assert (first.attempt, second.attempt) == (1, 2)
    (prev,) = second.previous_attempts
    assert (prev.attempt, prev.outcome, prev.accepted) == (1, "changed", False)
    assert prev.explanation == "try multiplication"
    assert prev.files_changed == [OPS_PY]
    assert prev.rejection_reason.startswith("target still failing")
    # After rollback the failure shown to the fixer is the original one again.
    assert second.failure_message == first.failure_message


@pytest.mark.parametrize("max_attempts", [1, 3])
def test_no_changes_is_unfixable_after_max_attempts(run, max_attempts: int) -> None:
    fixer = ScriptedFixer({SUBTRACT: [("noop",)] * 5})
    result = run(fixer, [SUBTRACT], max_attempts=max_attempts)

    outcome = _only(result)
    assert outcome.status == OutcomeStatus.UNFIXABLE
    assert outcome.attempts == max_attempts
    assert [r.attempt for r in fixer.requests] == list(range(1, max_attempts + 1))
    assert all(r.max_attempts == max_attempts for r in fixer.requests)
    assert outcome.reason == f"still failing after {max_attempts} attempt(s): no changes made"
    assert result.unfixable == [outcome]


def test_wrong_edits_every_time_is_unfixable_and_leave_no_diff(run) -> None:
    fixer = ScriptedFixer({SUBTRACT: [WRONG_SUBTRACT] * 2})
    result = run(fixer, [SUBTRACT], max_attempts=2)

    outcome = _only(result)
    assert (outcome.status, outcome.attempts) == (OutcomeStatus.UNFIXABLE, 2)
    assert "still failing after 2 attempt(s): target still failing" in outcome.reason
    assert outcome.files_changed == []
    assert result.diff == ""


def test_fix_breaking_already_passing_test_is_rejected(run) -> None:
    fixer = ScriptedFixer({SUBTRACT: [BREAK_ADD_AND_FIX_SUBTRACT, FIX_SUBTRACT]})
    result = run(fixer, [SUBTRACT, ADD])

    sub, add = result.tests
    assert (sub.status, sub.attempts) == (OutcomeStatus.FIXED, 2)
    assert add.status == OutcomeStatus.ALREADY_PASSING
    (prev,) = fixer.requests[1].previous_attempts
    assert prev.rejection_reason == f"broke previously passing tests: {ADD}"
    assert "a - b\n" in result.diff and ADD_BODY.splitlines()[0] not in result.diff


def test_fix_breaking_earlier_fixed_test_is_rejected(run) -> None:
    revert_divide = (
        "edits",
        [(OPS_PY, DIVIDE_FIX, DIVIDE_BUG), (OPS_PY, SUBTRACT_BUG, SUBTRACT_FIX)],
        "oops",
    )
    fixer = ScriptedFixer({DIV_ZERO: [FIX_DIVIDE], SUBTRACT: [revert_divide]})
    result = run(fixer, [DIV_ZERO, SUBTRACT], max_attempts=1)

    div, sub = result.tests
    assert div.status == OutcomeStatus.FIXED
    assert sub.status == OutcomeStatus.UNFIXABLE
    assert f"broke previously passing tests: {DIV_ZERO}" in sub.reason
    assert DIVIDE_FIX.splitlines()[0] in result.diff
    assert SUBTRACT_FIX not in result.diff


def test_fix_for_one_test_also_fixes_another(run) -> None:
    fixer = ScriptedFixer({SUBTRACT: [FIX_SUBTRACT_AND_DIVIDE]})
    result = run(fixer, [SUBTRACT, DIV_ZERO])

    sub, div = result.tests
    assert (sub.status, sub.attempts) == (OutcomeStatus.FIXED, 1)
    assert div.status == OutcomeStatus.FIXED
    assert div.reason == f"fixed by the fix for {SUBTRACT}"
    assert (div.attempts, div.files_changed) == (0, [OPS_PY])
    assert fixer.calls_for(DIV_ZERO) == []


@pytest.mark.parametrize(
    ("action", "reason"),
    [
        (DELETE_SUBTRACT_TEST, "test_removed"),
        (
            ("edit", TEST_OPS_PY, SUBTRACT_TEST, "@pytest.mark.skip\n" + SUBTRACT_TEST, "skip"),
            "skip_added",
        ),
    ],
)
def test_deleting_or_skipping_target_is_rejected(run, action, reason: str) -> None:
    # Slice 5: the patch checker rejects these before verification runs.
    result = run(ScriptedFixer({SUBTRACT: [action]}), [SUBTRACT], max_attempts=1)
    outcome = _only(result)
    assert outcome.status == OutcomeStatus.UNFIXABLE
    assert outcome.reason.startswith("still failing after 1 attempt(s): integrity check failed:")
    assert f"[{reason}]" in outcome.reason
    assert result.diff == ""


def test_deleting_another_requested_test_is_rejected(run) -> None:
    delete_and_fix_mean = (
        "edits",
        [(TEST_OPS_PY, SUBTRACT_TEST, ""), (OPS_PY, MEAN_BUG, MEAN_FIX)],
        "x",
    )
    fixer = ScriptedFixer({MEAN: [delete_and_fix_mean]})
    result = run(fixer, [MEAN, SUBTRACT], max_attempts=1)
    mean = _by_name(result)[MEAN]
    assert mean.status == OutcomeStatus.UNFIXABLE
    assert "integrity check failed:" in mean.reason  # slice 5: caught before verification
    assert "[test_removed]" in mean.reason and "test_subtract" in mean.reason


def test_verify_reruns_all_requested_tests(run, runners: FakeRunnerFactory) -> None:
    run(ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]}), [SUBTRACT, ADD, DIV_OK])
    (runner,) = runners.runners
    assert runner.runs == [[SUBTRACT, ADD, DIV_OK]] * 2  # initial run + one verify


# ---- fixer giving up or crashing ------------------------------------------------------------


def test_fixer_reports_unfixable_and_its_edits_are_rolled_back(run) -> None:
    gives_up = ("edit_then_unfixable", OPS_PY, MEAN_BUG, "return 0", "needs a product decision")
    fixer = ScriptedFixer({MEAN: [gives_up], SUBTRACT: [FIX_SUBTRACT]})
    result = run(fixer, [MEAN, SUBTRACT])

    mean, sub = result.tests
    assert mean.status == OutcomeStatus.UNFIXABLE
    assert mean.reason == "needs a product decision"
    assert (mean.attempts, mean.files_changed) == (1, [])
    assert sub.status == OutcomeStatus.FIXED
    assert "return 0" not in result.diff
    assert "+    return a - b" in result.diff


def test_fixer_raising_counts_as_attempt_and_partial_edits_are_dropped(run) -> None:
    crash = ("edit_then_raise", OPS_PY, SUBTRACT_BUG, "return a * b", "boom")
    fixer = ScriptedFixer({SUBTRACT: [crash, FIX_SUBTRACT]})
    result = run(fixer, [SUBTRACT])

    outcome = _only(result)
    assert (outcome.status, outcome.attempts) == (OutcomeStatus.FIXED, 2)
    (prev,) = fixer.requests[1].previous_attempts
    assert (prev.outcome, prev.accepted) == ("no_change", False)
    assert prev.rejection_reason == "fixer error: boom"
    assert "a * b" not in result.diff


def test_fixer_always_raising_is_unfixable(run) -> None:
    result = run(ScriptedFixer({SUBTRACT: [("raise", "boom")] * 5}), [SUBTRACT], max_attempts=2)
    outcome = _only(result)
    assert (outcome.status, outcome.attempts) == (OutcomeStatus.UNFIXABLE, 2)
    assert outcome.reason == "still failing after 2 attempt(s): fixer error: boom"


def test_files_changed_come_from_git_not_the_fixer(run) -> None:
    class Liar:
        def fix(self, request: FixRequest) -> FixAttempt:
            repo = Path(request.repo_path)
            ops = repo / OPS_PY
            ops.write_text(ops.read_text().replace(SUBTRACT_BUG, SUBTRACT_FIX), encoding="utf-8")
            (repo / "src" / "calc" / "brand_new.py").write_text("X = 1\n", encoding="utf-8")
            return FixAttempt(
                node_id="wrong",
                attempt=99,
                outcome="no_change",
                files_changed=["claimed.py"],
            )

    result = run(Liar(), [SUBTRACT])
    outcome = _only(result)
    assert outcome.status == OutcomeStatus.FIXED
    assert outcome.files_changed == ["src/calc/brand_new.py", OPS_PY]
    assert "+X = 1" in result.diff


# ---- checkpoints and the final diff ---------------------------------------------------------


def test_final_diff_has_only_accepted_fixes_and_checkpoints_exist(
    run, tmp_path: Path, sample_remote: SampleRemote
) -> None:
    gives_up = ("edit_then_unfixable", OPS_PY, MEAN_BUG, "return 0", "unclear")
    fixer = ScriptedFixer(
        {SUBTRACT: [WRONG_SUBTRACT, FIX_SUBTRACT], DIV_ZERO: [FIX_DIVIDE], MEAN: [gives_up]}
    )
    result = run(fixer, [SUBTRACT, MEAN, DIV_ZERO], keep_workspace=True)

    assert [t.status for t in result.tests] == [
        OutcomeStatus.FIXED,
        OutcomeStatus.UNFIXABLE,
        OutcomeStatus.FIXED,
    ]
    assert _checkpoints(tmp_path, sample_remote) == [
        f"ci-fix: fix {SUBTRACT} (attempt 2)",
        f"ci-fix: fix {DIV_ZERO} (attempt 1)",
    ]
    repo = run_dir_for(tmp_path) / "repo"
    assert git("status", "--porcelain", cwd=repo) == ""
    assert git("rev-parse", "--abbrev-ref", "HEAD", cwd=repo) == f"ci-fix/pr-{SAMPLE_PR}"
    assert "+    return a - b" in result.diff
    assert '+        raise ValueError("division by zero")' in result.diff
    assert "a * b" not in result.diff and "return 0" not in result.diff


# ---- ordering, names, no fix needed ---------------------------------------------------------


def test_tests_are_fixed_one_at_a_time_in_user_order(run) -> None:
    fixer = ScriptedFixer(
        {MEAN: [FIX_MEAN], SUBTRACT: [WRONG_SUBTRACT, FIX_SUBTRACT], DIV_ZERO: [FIX_DIVIDE]}
    )
    result = run(fixer, [MEAN, SUBTRACT, DIV_ZERO])
    assert all(t.status == OutcomeStatus.FIXED for t in result.tests)
    assert [(r.node_id, r.attempt) for r in fixer.requests] == [
        (MEAN, 1),
        (SUBTRACT, 1),
        (SUBTRACT, 2),
        (DIV_ZERO, 1),
    ]


@pytest.mark.parametrize("name", [ADD, SKIPPED])
def test_passing_or_skipped_is_already_passing(run, name: str) -> None:
    fixer = ScriptedFixer()
    result = run(fixer, [name])
    outcome = _only(result)
    assert (outcome.status, outcome.attempts) == (OutcomeStatus.ALREADY_PASSING, 0)
    assert fixer.requests == []
    assert result.diff == ""


def test_ambiguous_bare_name(run) -> None:
    fixer = ScriptedFixer()
    outcome = _only(run(fixer, ["test_add"]))
    assert (outcome.status, outcome.requested_name) == (OutcomeStatus.AMBIGUOUS, "test_add")
    assert fixer.requests == []


@pytest.mark.parametrize("name", ["test_does_not_exist", f"{TEST_OPS_PY}::test_does_not_exist"])
def test_unknown_name_not_found(run, name: str) -> None:
    fixer = ScriptedFixer()
    outcome = _only(run(fixer, [name]))
    assert (outcome.status, outcome.requested_name) == (OutcomeStatus.NOT_FOUND, name)
    assert fixer.requests == []


def test_multiple_tests_mixed_outcomes_keep_user_order(run) -> None:
    fixer = ScriptedFixer(
        {
            SUBTRACT: [FIX_SUBTRACT],
            DIV_ZERO: [FIX_DIVIDE],
            MEAN: [("noop",)] * 5,
            FIXTURE_ERROR: [("unfixable", "fixture is broken on purpose")],
        }
    )
    names = [
        MEAN,
        "test_nope",
        SUBTRACT,
        ADD,
        FIXTURE_ERROR,
        "test_add",
        DIV_ZERO,
        "test_subtract",  # bare name resolving to the same node id as SUBTRACT
    ]
    result = run(fixer, names, max_attempts=2)

    assert [t.requested_name for t in result.tests] == names
    by_name = _by_name(result)
    assert (by_name[MEAN].status, by_name[MEAN].attempts) == (OutcomeStatus.UNFIXABLE, 2)
    assert by_name["test_nope"].status == OutcomeStatus.NOT_FOUND
    assert by_name[SUBTRACT].status == OutcomeStatus.FIXED
    assert by_name[ADD].status == OutcomeStatus.ALREADY_PASSING
    assert by_name[FIXTURE_ERROR].status == OutcomeStatus.UNFIXABLE
    assert "fixture is broken" in by_name[FIXTURE_ERROR].reason
    assert by_name["test_add"].status == OutcomeStatus.AMBIGUOUS
    assert by_name[DIV_ZERO].status == OutcomeStatus.FIXED
    assert by_name["test_subtract"].status == OutcomeStatus.FIXED
    assert by_name["test_subtract"].node_id == SUBTRACT

    assert {t.requested_name for t in result.fixed} == {SUBTRACT, DIV_ZERO, "test_subtract"}
    assert len(fixer.calls_for(MEAN)) == 2
    assert len(fixer.calls_for(SUBTRACT)) == 1  # two names, one node id -> fixed once
    assert len(fixer.calls_for(FIXTURE_ERROR)) == 1
    assert fixer.calls_for(ADD) == []
    for name in names:
        assert name in result.summary


def test_noop_fixer_marks_failing_tests_unfixable(run) -> None:
    sub, add = run(NoOpFixer(), [SUBTRACT, ADD]).tests
    assert sub.status == OutcomeStatus.UNFIXABLE
    assert "No fixer configured" in sub.reason
    assert add.status == OutcomeStatus.ALREADY_PASSING


# ---- input validation -----------------------------------------------------------------------


def test_empty_failing_tests_raises(tmp_path: Path, sample_remote: SampleRemote) -> None:
    deps = make_deps(tmp_path, ScriptedFixer(), remote=sample_remote)
    with pytest.raises(ValueError):
        fix_failing_tests(REPO_URL, SAMPLE_PR, [], deps=deps)
    assert not (tmp_path / "ws").exists() or not any((tmp_path / "ws").iterdir())


@pytest.mark.parametrize("pr", [0, -1])
def test_non_positive_pr_raises(tmp_path: Path, sample_remote: SampleRemote, pr: int) -> None:
    deps = make_deps(tmp_path, ScriptedFixer(), remote=sample_remote)
    with pytest.raises(ValueError):
        fix_failing_tests(REPO_URL, pr, [SUBTRACT], deps=deps)


# ---- workspace cleanup ----------------------------------------------------------------------


def test_workspace_removed_after_success(run, tmp_path: Path) -> None:
    run(ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]}), [SUBTRACT])
    assert run_dirs_for(tmp_path) == []


def test_workspace_kept_with_keep_workspace(run, tmp_path: Path) -> None:
    run(ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]}), [SUBTRACT], keep_workspace=True)
    repo = run_dir_for(tmp_path) / "repo"
    assert "return a - b" in (repo / OPS_PY).read_text(encoding="utf-8")


@pytest.mark.parametrize("keep", [False, True])
def test_workspace_after_step_raises(
    tmp_path: Path, sample_remote: SampleRemote, keep: bool
) -> None:
    def broken_env(repo_path: Path, venv_dir: Path, settings):
        assert Path(repo_path).is_dir()  # setup ran, so there is something to clean up
        raise TestEnvError("uv venv failed")

    deps = make_deps(
        tmp_path, ScriptedFixer(), remote=sample_remote, env_factory=broken_env, keep_workspace=keep
    )
    with pytest.raises(TestEnvError, match="uv venv failed"):
        fix_failing_tests(REPO_URL, SAMPLE_PR, [SUBTRACT], deps=deps)
    assert len(run_dirs_for(tmp_path)) == int(keep)


def test_workspace_removed_when_setup_fails(tmp_path: Path, sample_remote: SampleRemote) -> None:
    deps = make_deps(tmp_path, ScriptedFixer(), remote=sample_remote)
    with pytest.raises(GitError):
        fix_failing_tests(REPO_URL, 99, [SUBTRACT], deps=deps)  # no refs/pull/99/head
    assert run_dirs_for(tmp_path, 99) == []


# ---- per-run state --------------------------------------------------------------------------


def test_deps_are_immutable_and_reusable(run, tmp_path, sample_remote, runners) -> None:
    deps = make_deps(
        tmp_path,
        ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]}),
        remote=sample_remote,
        runner_factory=runners,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        deps.fixer = NoOpFixer()  # type: ignore[misc]
    first = fix_failing_tests(REPO_URL, SAMPLE_PR, [SUBTRACT], deps=deps)
    second = fix_failing_tests(REPO_URL, SAMPLE_PR, [ADD], deps=deps)
    assert _only(first).status == OutcomeStatus.FIXED
    assert _only(second).status == OutcomeStatus.ALREADY_PASSING


def test_concurrent_runs_with_shared_deps_do_not_interfere(
    tmp_path: Path, sample_remote: SampleRemote, runners: FakeRunnerFactory
) -> None:
    fixer = StatelessFixer({SUBTRACT: FIX_SUBTRACT, MEAN: FIX_MEAN})
    deps = make_deps(tmp_path, fixer, remote=sample_remote, runner_factory=runners)
    jobs = {SAMPLE_PR: [SUBTRACT, ADD], OTHER_PR: [MEAN]}
    results: dict[int, FixResult] = {}
    errors: list[BaseException] = []
    barrier = threading.Barrier(len(jobs))

    def work(pr: int) -> None:
        try:
            barrier.wait(timeout=10)
            results[pr] = fix_failing_tests(REPO_URL, pr, jobs[pr], deps=deps)
        except BaseException as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(pr,)) for pr in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert errors == []

    a, b = results[SAMPLE_PR], results[OTHER_PR]
    assert [t.status for t in a.tests] == [OutcomeStatus.FIXED, OutcomeStatus.ALREADY_PASSING]
    assert [t.status for t in b.tests] == [OutcomeStatus.FIXED]
    assert (a.branch, b.branch) == (f"ci-fix/pr-{SAMPLE_PR}", f"ci-fix/pr-{OTHER_PR}")
    assert "a - b" in a.diff and MEAN_FIX not in a.diff
    assert MEAN_FIX in b.diff and "a - b" not in b.diff
    assert run_dirs_for(tmp_path, SAMPLE_PR) == []
    assert run_dirs_for(tmp_path, OTHER_PR) == []


# ---- logging --------------------------------------------------------------------------------


def test_info_logs_show_pipeline_steps(run, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="ci_fix")
    run(ScriptedFixer({SUBTRACT: [WRONG_SUBTRACT, FIX_SUBTRACT]}), [SUBTRACT])
    text = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.INFO)
    for prefix in ("[setup", "[fix", "[verify", "[finalize"):
        assert prefix in text, prefix
    assert "rejected: target still failing" in text


def test_unfixable_logged_as_warning(run, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="ci_fix")
    run(ScriptedFixer({SUBTRACT: [("unfixable", "nope")]}), [SUBTRACT])
    assert any(r.levelno >= logging.WARNING and SUBTRACT in r.getMessage() for r in caplog.records)


# ---- build/test artifacts -------------------------------------------------------------------


def _egg_info_env(repo_path: Path, venv_dir: Path, settings):
    """Like an editable install: leaves ``pkg.egg-info/`` in the checkout."""
    (Path(repo_path) / "pkg.egg-info").mkdir()
    (Path(repo_path) / "pkg.egg-info" / "PKG-INFO").write_text("Name: pkg\n")
    return fake_env_factory(repo_path, venv_dir, settings)


def test_setup_and_test_artifacts_never_reach_a_fix(
    run, tmp_path: Path, runners: FakeRunnerFactory
) -> None:
    runners.artifacts = True
    fixer = ScriptedFixer({SUBTRACT: [WRONG_SUBTRACT, FIX_SUBTRACT]})
    result = run(fixer, [SUBTRACT], env_factory=_egg_info_env, keep_workspace=True)

    outcome = _only(result)
    assert (outcome.status, outcome.files_changed) == (OutcomeStatus.FIXED, [OPS_PY])
    assert fixer.requests[1].previous_attempts[0].files_changed == [OPS_PY]
    for artifact in ("egg-info", ".coverage", "out/run-"):
        assert artifact not in result.diff
    repo = run_dir_for(tmp_path) / "repo"
    assert git("show", "--name-only", "--format=", "HEAD", cwd=repo).split() == [OPS_PY]
    # Artifacts are ignored, not deleted (rollback must not remove them either).
    assert (repo / "pkg.egg-info" / "PKG-INFO").is_file()
    assert (repo / ".coverage").is_file()
    assert (repo / "out" / "run-2.log").is_file()
    assert git("status", "--porcelain", cwd=repo) == ""


@pytest.mark.parametrize("script", [[("noop",)], [WRONG_SUBTRACT, ("noop",)]])
def test_noop_fixer_with_artifacts_is_no_changes(run, runners: FakeRunnerFactory, script) -> None:
    # Artifacts from setup and the initial run, and (2nd case) from attempt 1's verify run,
    # must not count as changes.
    runners.artifacts = True
    fixer = ScriptedFixer({SUBTRACT: script})
    result = run(fixer, [SUBTRACT], env_factory=_egg_info_env, max_attempts=len(script))
    outcome = _only(result)
    assert outcome.status == OutcomeStatus.UNFIXABLE
    assert outcome.reason == f"still failing after {len(script)} attempt(s): no changes made"
    assert result.diff == ""


# ---- flaky tests ----------------------------------------------------------------------------


def test_flaky_passing_test_does_not_reject_a_good_fix(
    run, runners: FakeRunnerFactory, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="ci_fix")
    runners.rules[ADD] = flaky(runners.rules[ADD], {2: False})  # fails only in the verify run
    result = run(ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]}), [SUBTRACT, ADD])

    sub, add = result.tests
    assert (sub.status, sub.attempts) == (OutcomeStatus.FIXED, 1)
    assert add.status == OutcomeStatus.ALREADY_PASSING
    (runner,) = runners.runners
    assert runner.runs == [[SUBTRACT, ADD], [SUBTRACT, ADD], [ADD]]
    assert f"{ADD} passed on re-run; treating as flaky" in caplog.text


def test_really_broken_test_is_rejected_after_re_run(run, runners: FakeRunnerFactory) -> None:
    fixer = ScriptedFixer({SUBTRACT: [BREAK_ADD_AND_FIX_SUBTRACT]})
    result = run(fixer, [SUBTRACT, ADD], max_attempts=1)
    assert result.tests[0].status == OutcomeStatus.UNFIXABLE
    (runner,) = runners.runners
    assert runner.runs[-1] == [ADD]  # re-run once before rejecting


def test_side_effect_fix_is_confirmed_by_re_run(run, runners: FakeRunnerFactory) -> None:
    run(ScriptedFixer({SUBTRACT: [FIX_SUBTRACT_AND_DIVIDE]}), [SUBTRACT, DIV_ZERO])
    (runner,) = runners.runners
    assert runner.runs[-1] == [DIV_ZERO]


def test_flaky_side_effect_fix_is_not_counted(
    run, runners: FakeRunnerFactory, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="ci_fix")
    runners.rules[DIV_ZERO] = flaky(runners.rules[DIV_ZERO], {2: True})  # passes once only
    fixer = ScriptedFixer({SUBTRACT: [FIX_SUBTRACT], DIV_ZERO: [FIX_DIVIDE]})
    result = run(fixer, [SUBTRACT, DIV_ZERO])

    sub, div = result.tests
    assert sub.status == OutcomeStatus.FIXED
    assert (div.status, div.attempts, div.reason) == (OutcomeStatus.FIXED, 1, "")
    assert len(fixer.calls_for(DIV_ZERO)) == 1
    assert f"{DIV_ZERO} failed on re-run; treating as flaky" in caplog.text


# ---- rejection reasons ----------------------------------------------------------------------


def test_initially_skipped_test_that_now_fails_is_a_regression(
    run, runners: FakeRunnerFactory
) -> None:
    runners.rules[SKIPPED] = marker_rule(runners.rules[SKIPPED], {"# BREAK": False})
    fix_and_break = ("edit", OPS_PY, SUBTRACT_BUG, SUBTRACT_FIX + "  # BREAK", "x")
    result = run(ScriptedFixer({SUBTRACT: [fix_and_break]}), [SUBTRACT, SKIPPED], max_attempts=1)
    sub, skipped = result.tests
    assert sub.status == OutcomeStatus.UNFIXABLE
    assert (
        sub.reason == f"still failing after 1 attempt(s): broke previously passing tests: {SKIPPED}"
    )
    assert skipped.status == OutcomeStatus.ALREADY_PASSING


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (TestStatus.NOT_FOUND, f"tests no longer collected after fix: {DIV_ZERO}"),
        (TestStatus.SKIPPED, f"tests skipped after fix: {DIV_ZERO}"),
    ],
)
def test_pending_test_vanishing_or_skipped_is_rejected(
    run, runners: FakeRunnerFactory, status: TestStatus, expected: str
) -> None:
    runners.rules[DIV_ZERO] = marker_rule(runners.rules[DIV_ZERO], {"# HIDE": status})
    fix_and_hide = ("edit", OPS_PY, SUBTRACT_BUG, SUBTRACT_FIX + "  # HIDE", "x")
    result = run(ScriptedFixer({SUBTRACT: [fix_and_hide]}), [SUBTRACT, DIV_ZERO], max_attempts=1)
    sub = result.tests[0]
    assert sub.status == OutcomeStatus.UNFIXABLE
    assert expected in sub.reason


def test_target_failing_to_error_is_still_failing(run, runners: FakeRunnerFactory) -> None:
    runners.rules[SUBTRACT] = marker_rule(runners.rules[SUBTRACT], {"# ERR": TestStatus.ERROR})
    to_error = ("edit", OPS_PY, SUBTRACT_BUG, SUBTRACT_BUG + "  # ERR", "x")
    result = run(ScriptedFixer({SUBTRACT: [to_error]}), [SUBTRACT], max_attempts=1)
    outcome = _only(result)
    assert outcome.status == OutcomeStatus.UNFIXABLE
    assert (
        outcome.reason
        == f"still failing after 1 attempt(s): target still failing: error: {SUBTRACT}"
    )


def test_worst_case_attempts_fit_the_recursion_limit(run) -> None:
    useless = ("edit", OPS_PY, '"""Basic arithmetic.', '"""Basic arithmetic (edited).', "x")
    tests = [SUBTRACT, DIV_ZERO, MEAN]
    fixer = ScriptedFixer({nid: [useless] * 5 for nid in tests})
    result = run(fixer, tests, max_attempts=5)
    assert [(t.status, t.attempts) for t in result.tests] == [(OutcomeStatus.UNFIXABLE, 5)] * 3
    assert len(fixer.requests) == 15


# ---- concurrent runs on the same PR ---------------------------------------------------------


def test_concurrent_runs_on_same_pr_use_separate_workspaces(
    tmp_path: Path, sample_remote: SampleRemote, runners: FakeRunnerFactory
) -> None:
    fixer = StatelessFixer({SUBTRACT: FIX_SUBTRACT, MEAN: FIX_MEAN})
    deps = make_deps(tmp_path, fixer, remote=sample_remote, runner_factory=runners)
    jobs = [[SUBTRACT], [MEAN]]
    results: list[FixResult | None] = [None, None]
    errors: list[BaseException] = []
    barrier = threading.Barrier(len(jobs))

    def work(i: int) -> None:
        try:
            barrier.wait(timeout=10)
            results[i] = fix_failing_tests(REPO_URL, SAMPLE_PR, jobs[i], deps=deps)
        except BaseException as exc:  # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(len(jobs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert errors == []
    a, b = results
    assert a is not None and b is not None
    assert _only(a).status == _only(b).status == OutcomeStatus.FIXED
    assert "a - b" in a.diff and MEAN_FIX not in a.diff
    assert MEAN_FIX in b.diff and "a - b" not in b.diff
    repo_paths = {r.repo_path for r in runners.runners}
    assert len(repo_paths) == 2  # two separate workspaces
    assert run_dirs_for(tmp_path) == []  # both cleaned up


def test_fatal_fixer_error_stops_run_without_burning_attempts(run, tmp_path: Path) -> None:
    from ci_fix.models import FixerFatalError

    class _Fatal:
        calls = 0

        def fix(self, request):
            _Fatal.calls += 1
            raise FixerFatalError("Claude API rejected the request: credit balance is too low")

    with pytest.raises(FixerFatalError, match="credit balance"):
        run(_Fatal(), [SUBTRACT, DIV_ZERO], max_attempts=3)
    assert _Fatal.calls == 1  # no further attempts, no other tests tried
    assert run_dirs_for(tmp_path) == []  # workspace still cleaned up
