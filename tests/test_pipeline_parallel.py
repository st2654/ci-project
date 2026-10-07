"""Slice 7: parallel fixing of independent failures (worktrees), on the ``FakeRunner``.

Two seeded bugs in different modules (``ops.py``/``test_ops.py`` and
``strings.py``/``test_strings.py``) are independent; scripted tracebacks tell the triage
step which files each failure touches.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from conftest import git
from parallel_helpers import (
    DELETE_SHOUT_TEST,
    FIX_SHOUT,
    RULES,
    SHOUT,
    SHOUT_BUG,
    SHOUT_FIX,
    STRINGS_PY,
    TB_SHOUT,
    TB_SHOUT_VIA_OPS,
    TB_SUBTRACT,
    ConcurrentFixer,
    TracebackRunnerFactory,
    make_two_module_remote,
    max_concurrency,
    overlapped,
)
from pipeline_helpers import (
    FIX_SUBTRACT,
    OPS_PY,
    REPO_URL,
    SAMPLE_PR,
    SUBTRACT,
    SUBTRACT_BUG,
    SUBTRACT_FIX,
    SampleRemote,
    make_deps,
    run_dir_for,
    run_dirs_for,
)

from ci_fix.models import FixerFatalError, FixResult, OutcomeStatus, TestOutcome
from ci_fix.pipeline import fix_failing_tests

DELAY = 0.4  # long enough that serial calls could never overlap by accident

README = "README.md"
README_TEST = "tests/test_errors.py::test_readme"
README_MARK = "Fixed by the readme test."
FIX_README = ("edit", README, "# ", f"{README_MARK}\n\n# ", "readme lacked the marker")


@pytest.fixture(scope="module")
def remote(tmp_path_factory: pytest.TempPathFactory) -> SampleRemote:
    return make_two_module_remote(tmp_path_factory.mktemp("two-module"))


@pytest.fixture
def run(tmp_path: Path, remote: SampleRemote):
    def _run(
        fixer,
        tests: list[str],
        *,
        tracebacks: dict[str, str] | None = None,
        rules=None,
        **overrides,
    ) -> FixResult:
        factory = TracebackRunnerFactory(
            rules=dict(rules or RULES),
            tracebacks=dict(tracebacks or {SUBTRACT: TB_SUBTRACT, SHOUT: TB_SHOUT}),
        )
        deps = make_deps(tmp_path, fixer, remote=remote, runner_factory=factory, **overrides)
        return fix_failing_tests(REPO_URL, SAMPLE_PR, tests, deps=deps)

    return _run


def _by_id(result: FixResult) -> dict[str, TestOutcome]:
    return {t.node_id: t for t in result.tests}


def _checkpoints(tmp_path: Path, remote: SampleRemote) -> list[str]:
    repo = run_dir_for(tmp_path) / "repo"
    log = git("log", "--format=%s", "--reverse", f"{remote.pr_sha}..HEAD", cwd=repo)
    return log.splitlines()


def _worktrees(repo: Path) -> list[Path]:
    out = git("worktree", "list", "--porcelain", cwd=repo)
    return [
        Path(line[len("worktree ") :]) for line in out.splitlines() if line.startswith("worktree ")
    ]


def _is_under(path: Path, parent: Path) -> bool:
    return parent.resolve() in path.resolve().parents


def _info(caplog: pytest.LogCaptureFixture) -> str:
    return "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.INFO)


# ---- the happy path -------------------------------------------------------------------------


def test_independent_failures_are_fixed_in_parallel_worktrees(
    run, tmp_path: Path, remote: SampleRemote, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="ci_fix")
    fixer = ConcurrentFixer({SUBTRACT: [FIX_SUBTRACT], SHOUT: [FIX_SHOUT]}, delay=DELAY)
    result = run(fixer, [SUBTRACT, SHOUT], keep_workspace=True)

    outcomes = _by_id(result)
    assert (outcomes[SUBTRACT].status, outcomes[SUBTRACT].attempts) == (OutcomeStatus.FIXED, 1)
    assert (outcomes[SHOUT].status, outcomes[SHOUT].attempts) == (OutcomeStatus.FIXED, 1)
    assert outcomes[SUBTRACT].files_changed == [OPS_PY]
    assert outcomes[SHOUT].files_changed == [STRINGS_PY]

    # Both fixer calls ran at the same time, on different threads.
    sub_call, shout_call = sorted(fixer.calls, key=lambda c: c.node_id != SUBTRACT)
    assert overlapped(sub_call, shout_call)
    assert sub_call.thread != shout_call.thread

    # Each ran in its own worktree under run_dir/worktrees/, never the main checkout.
    run_dir = run_dir_for(tmp_path)
    main = run_dir / "repo"
    for call in (sub_call, shout_call):
        assert call.repo_path.resolve() != main.resolve()
        assert _is_under(call.repo_path, run_dir / "worktrees")
    assert sub_call.repo_path.resolve() != shout_call.repo_path.resolve()

    # Both fixes merged into the main checkout as serial checkpoints, in pending order.
    assert f"+    {SUBTRACT_FIX}" in result.diff
    assert f"+    {SHOUT_FIX}" in result.diff
    assert _checkpoints(tmp_path, remote) == [
        f"ci-fix: fix {SUBTRACT} (attempt 1)",
        f"ci-fix: fix {SHOUT} (attempt 1)",
    ]
    assert SHOUT_FIX in (main / STRINGS_PY).read_text(encoding="utf-8")
    assert SUBTRACT_FIX in (main / OPS_PY).read_text(encoding="utf-8")

    # Worktrees are gone afterwards.
    assert [p.resolve() for p in _worktrees(main)] == [main.resolve()]
    assert "[parallel] round 1: fixing 2 test(s) in parallel" in _info(caplog)


def test_worktree_is_at_current_head_and_clean(run, remote: SampleRemote) -> None:
    seen: dict[str, tuple[str, str]] = {}

    class _Probe(ConcurrentFixer):
        def fix(self, request):
            repo = Path(request.repo_path)
            seen[request.node_id] = (
                git("rev-parse", "HEAD", cwd=repo),
                git("status", "--porcelain", "--untracked-files=no", cwd=repo),
            )
            return super().fix(request)

    fixer = _Probe({SUBTRACT: [FIX_SUBTRACT], SHOUT: [FIX_SHOUT]}, delay=DELAY)
    result = run(fixer, [SUBTRACT, SHOUT], keep_workspace=True)

    assert {t.status for t in result.tests} == {OutcomeStatus.FIXED}
    # Both candidates started from the PR head (HEAD before round 1), with a clean tree.
    assert seen[SUBTRACT] == (remote.pr_sha, "")
    assert seen[SHOUT] == (remote.pr_sha, "")


def test_max_parallel_workers_limits_concurrency(run, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="ci_fix")
    rules = {**RULES, README_TEST: lambda read: README_MARK in read(README)}
    tracebacks = {
        SUBTRACT: TB_SUBTRACT,
        SHOUT: TB_SHOUT,
        README_TEST: "tests/test_errors.py:20: in test_readme\nE   AssertionError\n",
    }
    fixer = ConcurrentFixer(
        {SUBTRACT: [FIX_SUBTRACT], SHOUT: [FIX_SHOUT], README_TEST: [FIX_README]}, delay=DELAY
    )
    result = run(
        fixer,
        [SUBTRACT, SHOUT, README_TEST],
        rules=rules,
        tracebacks=tracebacks,
        max_parallel_workers=2,
    )

    assert {t.node_id: t.status for t in result.tests} == {
        SUBTRACT: OutcomeStatus.FIXED,
        SHOUT: OutcomeStatus.FIXED,
        README_TEST: OutcomeStatus.FIXED,
    }
    assert max_concurrency(fixer.calls) == 2
    assert "[parallel] round 1: fixing 2 test(s) in parallel" in _info(caplog)


# ---- serial fallbacks -----------------------------------------------------------------------


def test_dependent_failures_are_fixed_one_after_another(
    run, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="ci_fix")
    fixer = ConcurrentFixer({SUBTRACT: [FIX_SUBTRACT], SHOUT: [FIX_SHOUT]}, delay=DELAY / 2)
    result = run(
        fixer,
        [SUBTRACT, SHOUT],
        tracebacks={SUBTRACT: TB_SUBTRACT, SHOUT: TB_SHOUT_VIA_OPS},
        keep_workspace=True,
    )

    assert {t.status for t in result.tests} == {OutcomeStatus.FIXED}
    first, second = sorted(fixer.calls, key=lambda c: c.start)
    assert (first.node_id, second.node_id) == (SUBTRACT, SHOUT)
    assert not overlapped(first, second)
    main = run_dir_for(tmp_path) / "repo"
    assert {c.repo_path.resolve() for c in fixer.calls} == {main.resolve()}
    assert "in parallel" not in _info(caplog)


def test_one_worker_is_serial_and_matches_parallel_results(
    tmp_path: Path, remote: SampleRemote, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="ci_fix")
    script = {SUBTRACT: [FIX_SUBTRACT], SHOUT: [FIX_SHOUT]}

    def go(workers: int, fixer: ConcurrentFixer, sub: str) -> FixResult:
        factory = TracebackRunnerFactory(rules=dict(RULES))
        deps = make_deps(
            tmp_path / sub,
            fixer,
            remote=remote,
            runner_factory=factory,
            max_parallel_workers=workers,
        )
        return fix_failing_tests(REPO_URL, SAMPLE_PR, [SUBTRACT, SHOUT], deps=deps)

    serial_fixer = ConcurrentFixer(script, delay=DELAY / 2)
    serial = go(1, serial_fixer, "serial")
    assert "in parallel" not in _info(caplog)
    first, second = sorted(serial_fixer.calls, key=lambda c: c.start)
    assert (first.node_id, second.node_id) == (SUBTRACT, SHOUT)
    assert not overlapped(first, second)
    assert {c.repo_path.name for c in serial_fixer.calls} == {"repo"}
    assert not any("worktrees" in c.repo_path.parts for c in serial_fixer.calls)

    parallel = go(4, ConcurrentFixer(script, delay=DELAY / 2), "parallel")

    def summary(r: FixResult):
        return [(t.node_id, t.status, t.attempts, t.files_changed, t.reason) for t in r.tests]

    assert summary(serial) == summary(parallel)
    assert serial.diff == parallel.diff


# ---- merging candidates ---------------------------------------------------------------------


def test_conflicting_patch_is_retried_sequentially_without_counting(
    run, tmp_path: Path, remote: SampleRemote, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="ci_fix")
    # SHOUT's first candidate also rewrites the subtract line, differently from SUBTRACT's.
    clash = (
        "edits",
        [(OPS_PY, SUBTRACT_BUG, "return -(b - a)"), (STRINGS_PY, SHOUT_BUG, SHOUT_FIX)],
        "fix shout and tidy subtract",
    )
    fixer = ConcurrentFixer({SUBTRACT: [FIX_SUBTRACT], SHOUT: [clash, FIX_SHOUT]}, delay=DELAY)
    result = run(fixer, [SUBTRACT, SHOUT], keep_workspace=True)

    outcomes = _by_id(result)
    assert (outcomes[SUBTRACT].status, outcomes[SUBTRACT].attempts) == (OutcomeStatus.FIXED, 1)
    assert (outcomes[SHOUT].status, outcomes[SHOUT].attempts) == (OutcomeStatus.FIXED, 1)
    shout_requests = fixer.calls_for(SHOUT)
    assert [r.attempt for r in shout_requests] == [1, 1]  # the conflict was not counted
    # The retry is sequential, in the main checkout.
    main = run_dir_for(tmp_path) / "repo"
    assert Path(shout_requests[1].repo_path).resolve() == main.resolve()

    assert f"+    {SUBTRACT_FIX}" in result.diff
    assert "-(b - a)" not in result.diff
    assert f"+    {SHOUT_FIX}" in result.diff
    assert _checkpoints(tmp_path, remote) == [
        f"ci-fix: fix {SUBTRACT} (attempt 1)",
        f"ci-fix: fix {SHOUT} (attempt 1)",
    ]
    assert any(
        "[parallel]" in line and "retrying sequentially" in line and SHOUT in line
        for line in _info(caplog).splitlines()
    )
    assert [p.resolve() for p in _worktrees(main)] == [main.resolve()]


def test_candidate_for_test_fixed_by_earlier_merge_is_discarded(
    run, tmp_path: Path, remote: SampleRemote
) -> None:
    # SHOUT secretly depends on SUBTRACT's fix (its traceback doesn't show it).
    rules = {**RULES, SHOUT: lambda read: SUBTRACT_FIX in read(OPS_PY)}
    fixer = ConcurrentFixer({SUBTRACT: [FIX_SUBTRACT], SHOUT: [FIX_SHOUT]}, delay=DELAY)
    result = run(fixer, [SUBTRACT, SHOUT], rules=rules, keep_workspace=True)

    outcomes = _by_id(result)
    assert (outcomes[SUBTRACT].status, outcomes[SUBTRACT].attempts) == (OutcomeStatus.FIXED, 1)
    assert outcomes[SHOUT].status == OutcomeStatus.FIXED
    assert outcomes[SHOUT].reason == f"fixed by the fix for {SUBTRACT}"
    assert len(fixer.calls_for(SHOUT)) == 1  # it did get a parallel candidate...
    assert SHOUT_FIX not in result.diff  # ...whose patch was discarded
    assert _checkpoints(tmp_path, remote) == [f"ci-fix: fix {SUBTRACT} (attempt 1)"]


def test_integrity_violation_in_parallel_candidate_is_rejected_and_retried(
    run, tmp_path: Path
) -> None:
    fixer = ConcurrentFixer(
        {SUBTRACT: [FIX_SUBTRACT], SHOUT: [DELETE_SHOUT_TEST, FIX_SHOUT]}, delay=DELAY
    )
    result = run(fixer, [SUBTRACT, SHOUT], keep_workspace=True)

    outcomes = _by_id(result)
    assert (outcomes[SUBTRACT].status, outcomes[SUBTRACT].attempts) == (OutcomeStatus.FIXED, 1)
    assert (outcomes[SHOUT].status, outcomes[SHOUT].attempts) == (OutcomeStatus.FIXED, 2)
    first, second = fixer.calls_for(SHOUT)
    assert "worktrees" in Path(first.repo_path).parts  # the bad candidate ran in parallel
    (prev,) = second.previous_attempts
    assert prev.accepted is False
    assert prev.rejection_reason.startswith("integrity check failed:")
    assert "test_removed" in prev.rejection_reason
    assert (
        "def test_shout" in (run_dir_for(tmp_path) / "repo" / "tests/test_strings.py").read_text()
    )


def test_fixer_error_in_one_worktree_counts_and_other_test_is_fixed(run, tmp_path: Path) -> None:
    fixer = ConcurrentFixer(
        {SUBTRACT: [FIX_SUBTRACT], SHOUT: [("raise", "boom"), FIX_SHOUT]}, delay=DELAY
    )
    result = run(fixer, [SUBTRACT, SHOUT], keep_workspace=True)

    outcomes = _by_id(result)
    assert (outcomes[SUBTRACT].status, outcomes[SUBTRACT].attempts) == (OutcomeStatus.FIXED, 1)
    assert (outcomes[SHOUT].status, outcomes[SHOUT].attempts) == (OutcomeStatus.FIXED, 2)
    first, second = fixer.calls_for(SHOUT)
    assert (first.attempt, second.attempt) == (1, 2)
    (prev,) = second.previous_attempts
    assert prev.rejection_reason == "fixer error: boom"
    main = run_dir_for(tmp_path) / "repo"
    assert [p.resolve() for p in _worktrees(main)] == [main.resolve()]


def test_fixer_error_every_time_is_unfixable_other_still_fixed(run) -> None:
    fixer = ConcurrentFixer(
        {SUBTRACT: [FIX_SUBTRACT], SHOUT: [("raise", "boom")] * 3}, delay=DELAY / 2
    )
    result = run(fixer, [SUBTRACT, SHOUT])

    outcomes = _by_id(result)
    assert outcomes[SUBTRACT].status == OutcomeStatus.FIXED
    assert (outcomes[SHOUT].status, outcomes[SHOUT].attempts) == (OutcomeStatus.UNFIXABLE, 3)
    assert outcomes[SHOUT].reason == "still failing after 3 attempt(s): fixer error: boom"
    assert SHOUT_FIX not in result.diff


def test_fatal_fixer_error_stops_run_and_removes_worktrees(run, tmp_path: Path) -> None:
    fixer = ConcurrentFixer(
        {SUBTRACT: [FIX_SUBTRACT], SHOUT: [("fatal", "credit balance is too low")]},
        delay=DELAY / 2,
    )
    with pytest.raises(FixerFatalError, match="credit balance"):
        run(fixer, [SUBTRACT, SHOUT], keep_workspace=True)

    assert len(fixer.calls_for(SHOUT)) == 1
    run_dir = run_dir_for(tmp_path)
    main = run_dir / "repo"
    assert [p.resolve() for p in _worktrees(main)] == [main.resolve()]
    worktrees_dir = run_dir / "worktrees"
    assert not worktrees_dir.exists() or list(worktrees_dir.iterdir()) == []


def test_fatal_fixer_error_without_keep_workspace_cleans_everything(run, tmp_path: Path) -> None:
    fixer = ConcurrentFixer({SUBTRACT: [FIX_SUBTRACT], SHOUT: [("fatal", "bad key")]})
    with pytest.raises(FixerFatalError, match="bad key"):
        run(fixer, [SUBTRACT, SHOUT])
    assert run_dirs_for(tmp_path) == []


def test_parallel_run_cleans_workspace_by_default(run, tmp_path: Path) -> None:
    fixer = ConcurrentFixer({SUBTRACT: [FIX_SUBTRACT], SHOUT: [FIX_SHOUT]})
    result = run(fixer, [SUBTRACT, SHOUT])
    assert {t.status for t in result.tests} == {OutcomeStatus.FIXED}
    assert run_dirs_for(tmp_path) == []
