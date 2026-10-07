"""Slice 6: full-suite regression check after source-changing fixes.

Fast graph-logic tests on a ``FakeRunner`` that also implements ``run_all`` over a declared
suite (statuses decided from the checkout's file contents). One end-to-end test on the real
pytest is at the bottom.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from conftest import git
from pipeline_helpers import (
    ADD,
    ADD_BODY,
    DIV_ZERO,
    DIVIDE_BUG,
    DIVIDE_FIX,
    ERRS_ADD,
    FIX_DIVIDE,
    FIX_SUBTRACT,
    FIXTURE_ERROR,
    MEAN,
    OPS_PY,
    REPO_URL,
    SAMPLE_PR,
    SAMPLE_RULES,
    SUBTRACT,
    SUBTRACT_BUG,
    SUBTRACT_FIX,
    TEST_OPS_PY,
    WRONG_SUBTRACT,
    FakeRunner,
    Read,
    Rule,
    SampleRemote,
    ScriptedFixer,
    flaky,
    make_deps,
    run_dir_for,
)

from ci_fix.config import Settings
from ci_fix.graph import RESTORE_FAILED_PREFIX
from ci_fix.models import FixResult, OutcomeStatus
from ci_fix.pipeline import fix_failing_tests
from ci_fix.tools.git import GitError, GitRepo
from ci_fix.tools.pytest_runner import TestResult, TestRunError, TestRunResult, TestStatus

BROKE = "broke other tests in the full suite"
SKIPPED_WARNING = "regression check skipped:"

# Suite-only tests (never requested): they exist only in the full-suite run.
EXTRA = "tests/test_extra.py::test_extra"
EXTRA_MARKER = "  # EXTRA"

BREAK_ADD_AND_FIX_SUBTRACT = (
    "edits",
    [
        (OPS_PY, ADD_BODY, ADD_BODY.replace("a + b\n", "a - b\n")),
        (OPS_PY, SUBTRACT_BUG, SUBTRACT_FIX),
    ],
    "subtract everywhere",
)
EXPECT_8 = "assert subtract(5, 3) == 8"
TEST_ONLY_FIX = (
    "edit",
    TEST_OPS_PY,
    "assert subtract(5, 3) == 2",
    EXPECT_8,
    "Root cause: the test expected the wrong value.\nTest change: subtract is documented to add.",
)


def _lenient_subtract(read: Read) -> bool | TestStatus:
    """Like the sample rule, but the test-only expectation change also passes."""
    tests = read(TEST_OPS_PY)
    if "def test_subtract" not in tests:
        return TestStatus.NOT_FOUND
    return SUBTRACT_FIX in read(OPS_PY) or EXPECT_8 in tests


# ---- fake runner with run_all ---------------------------------------------------------------


@dataclass
class SuiteRunner(FakeRunner):
    """``FakeRunner`` plus ``run_all``: the full suite is ``rules`` + ``suite_rules``.

    A test whose verdict is NOT_FOUND is omitted from ``run_all`` results (= missing).
    ``run_all_errors`` maps a 1-based ``run_all`` call number to an error to raise.
    """

    suite_rules: dict[str, Rule] = field(default_factory=dict)
    run_all_errors: dict[int, Exception] = field(default_factory=dict)
    run_all_calls: list[dict[str, Any]] = field(default_factory=list)

    def run_all(self, extra_args: Sequence[str] = (), timeout: float | None = None):
        self.run_all_calls.append(
            {"extra_args": list(extra_args), "timeout": timeout, "ops": self._read(OPS_PY)}
        )
        error = self.run_all_errors.get(len(self.run_all_calls))
        if error is not None:
            raise error
        results = {}
        for nid in {**self.rules, **self.suite_rules}:
            result = self._result(nid)
            if result.status != TestStatus.NOT_FOUND:
                results[nid] = result
        failed = any(r.status in (TestStatus.FAILED, TestStatus.ERROR) for r in results.values())
        return TestRunResult(results=results, exit_code=int(failed), duration=0.0, output_tail="")

    def _result(self, node_id: str) -> TestResult:
        rule = self.rules.get(node_id) or self.suite_rules.get(node_id)
        verdict = rule(self._read) if rule is not None else TestStatus.NOT_FOUND
        if isinstance(verdict, bool):
            verdict = TestStatus.PASSED if verdict else TestStatus.FAILED
        message = "" if verdict == TestStatus.PASSED else f"{verdict.value}: {node_id}"
        return TestResult(node_id=node_id, status=verdict, message=message, details=message)


@dataclass
class SuiteRunnerFactory:
    rules: dict[str, Rule] = field(default_factory=lambda: dict(SAMPLE_RULES))
    suite_rules: dict[str, Rule] = field(default_factory=dict)
    run_all_errors: dict[int, Exception] = field(default_factory=dict)
    runners: list[SuiteRunner] = field(default_factory=list)

    def __call__(self, repo_path: Path, python: Path, reports_dir: Path, settings: Settings):
        runner = SuiteRunner(
            Path(repo_path),
            self.rules,
            suite_rules=self.suite_rules,
            run_all_errors=self.run_all_errors,
        )
        self.runners.append(runner)
        return runner

    @property
    def runner(self) -> SuiteRunner:
        (runner,) = self.runners
        return runner


@pytest.fixture
def runners() -> SuiteRunnerFactory:
    return SuiteRunnerFactory()


@pytest.fixture
def run(tmp_path: Path, sample_remote: SampleRemote, runners: SuiteRunnerFactory):
    def _run(fixer, tests: list[str], **overrides) -> FixResult:
        deps = make_deps(tmp_path, fixer, remote=sample_remote, runner_factory=runners, **overrides)
        return fix_failing_tests(REPO_URL, SAMPLE_PR, tests, deps=deps)

    return _run


def _rejections(fixer: ScriptedFixer, node_id: str) -> list[str]:
    last = fixer.calls_for(node_id)[-1]
    return [p.rejection_reason for p in last.previous_attempts]


# ---- regressions ----------------------------------------------------------------------------


def test_source_fix_breaking_suite_test_is_rejected_then_correct_fix_accepted(
    run, runners: SuiteRunnerFactory
) -> None:
    fixer = ScriptedFixer({SUBTRACT: [BREAK_ADD_AND_FIX_SUBTRACT, FIX_SUBTRACT]})
    result = run(fixer, [SUBTRACT])

    (outcome,) = result.tests
    assert (outcome.status, outcome.attempts) == (OutcomeStatus.FIXED, 2)
    (reason,) = _rejections(fixer, SUBTRACT)
    assert reason.startswith(f"{BROKE}: ")
    assert ADD in reason and ERRS_ADD in reason
    assert SUBTRACT not in reason
    # Attempt 2's edit of SUBTRACT_BUG only applies if attempt 1 was rolled back.
    assert "a - b\n" in result.diff
    assert not any(line == "-    return a + b" for line in result.diff.splitlines())
    assert len(runners.runner.run_all_calls) == 3  # baseline, attempt 1, attempt 2


def test_rejected_regression_attempt_is_rolled_back_when_attempts_run_out(run) -> None:
    fixer = ScriptedFixer({SUBTRACT: [BREAK_ADD_AND_FIX_SUBTRACT]})
    result = run(fixer, [SUBTRACT], max_attempts=1)
    (outcome,) = result.tests
    assert outcome.status == OutcomeStatus.UNFIXABLE
    assert BROKE in outcome.reason
    assert result.diff == ""


def test_regression_reason_fed_to_next_attempt(run) -> None:
    fixer = ScriptedFixer({SUBTRACT: [BREAK_ADD_AND_FIX_SUBTRACT, FIX_SUBTRACT]})
    run(fixer, [SUBTRACT])
    first, second = fixer.requests
    (prev,) = second.previous_attempts
    assert (prev.attempt, prev.accepted) == (1, False)
    assert BROKE in prev.rejection_reason
    assert second.failure_message == first.failure_message  # rolled back


def test_suite_test_going_missing_is_a_regression(run, runners: SuiteRunnerFactory) -> None:
    runners.suite_rules[EXTRA] = lambda read: (
        TestStatus.NOT_FOUND if "# HIDE" in read(OPS_PY) else True
    )
    hide = ("edit", OPS_PY, SUBTRACT_BUG, SUBTRACT_FIX + "  # HIDE", "x")
    fixer = ScriptedFixer({SUBTRACT: [hide, FIX_SUBTRACT]})
    result = run(fixer, [SUBTRACT])
    assert result.tests[0].attempts == 2
    (reason,) = _rejections(fixer, SUBTRACT)
    assert BROKE in reason and EXTRA in reason


def test_suite_test_passing_to_error_is_a_regression(run, runners: SuiteRunnerFactory) -> None:
    runners.suite_rules[EXTRA] = lambda read: TestStatus.ERROR if "# ERR" in read(OPS_PY) else True
    err = ("edit", OPS_PY, SUBTRACT_BUG, SUBTRACT_FIX + "  # ERR", "x")
    fixer = ScriptedFixer({SUBTRACT: [err, FIX_SUBTRACT]})
    result = run(fixer, [SUBTRACT])
    assert result.tests[0].attempts == 2
    (reason,) = _rejections(fixer, SUBTRACT)
    assert BROKE in reason and EXTRA in reason


def test_more_than_ten_regressions_message_is_truncated(run, runners: SuiteRunnerFactory) -> None:
    extras = [f"tests/test_more.py::test_add_{i:02d}" for i in range(11)]
    for nid in extras:
        runners.suite_rules[nid] = SAMPLE_RULES[ADD]
    fixer = ScriptedFixer({SUBTRACT: [BREAK_ADD_AND_FIX_SUBTRACT, FIX_SUBTRACT]})
    run(fixer, [SUBTRACT])

    (reason,) = _rejections(fixer, SUBTRACT)
    assert reason.startswith(f"{BROKE}: ")
    regressed = [ADD, ERRS_ADD, *extras]  # 13
    listed = [nid for nid in regressed if nid in reason]
    assert len(listed) == 10
    assert "and 3 more" in reason


def test_exactly_ten_regressions_not_truncated(run, runners: SuiteRunnerFactory) -> None:
    extras = [f"tests/test_more.py::test_add_{i:02d}" for i in range(8)]
    for nid in extras:
        runners.suite_rules[nid] = SAMPLE_RULES[ADD]
    fixer = ScriptedFixer({SUBTRACT: [BREAK_ADD_AND_FIX_SUBTRACT, FIX_SUBTRACT]})
    run(fixer, [SUBTRACT])
    (reason,) = _rejections(fixer, SUBTRACT)
    assert all(nid in reason for nid in [ADD, ERRS_ADD, *extras])
    assert " more" not in reason


def test_flaky_regression_is_dropped_after_re_run(run, runners: SuiteRunnerFactory) -> None:
    # Evaluations of EXTRA: 1 = baseline (pass), 2 = after the fix (fail), 3 = re-run (pass).
    runners.suite_rules[EXTRA] = flaky(lambda read: True, {2: False})
    result = run(ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]}), [SUBTRACT])
    (outcome,) = result.tests
    assert (outcome.status, outcome.attempts) == (OutcomeStatus.FIXED, 1)
    assert "+    return a - b" in result.diff


def test_real_regression_is_re_run_before_rejecting(run, runners: SuiteRunnerFactory) -> None:
    calls = 0

    def counting(read: Read) -> bool:
        nonlocal calls
        calls += 1
        return "# BREAK" not in read(OPS_PY)

    runners.suite_rules[EXTRA] = counting
    brk = ("edit", OPS_PY, SUBTRACT_BUG, SUBTRACT_FIX + "  # BREAK", "x")
    result = run(ScriptedFixer({SUBTRACT: [brk]}), [SUBTRACT], max_attempts=1)
    assert result.tests[0].status == OutcomeStatus.UNFIXABLE
    assert calls == 3  # baseline, full run after the fix, one re-run


# ---- when the full suite runs ---------------------------------------------------------------


def test_test_only_fix_never_runs_full_suite(run, runners: SuiteRunnerFactory) -> None:
    runners.rules[SUBTRACT] = _lenient_subtract
    result = run(ScriptedFixer({SUBTRACT: [TEST_ONLY_FIX]}), [SUBTRACT])
    (outcome,) = result.tests
    assert (outcome.status, outcome.source_changed) == (OutcomeStatus.FIXED, False)
    assert runners.runner.run_all_calls == []
    assert result.preexisting_failures == []
    assert result.warnings == []


def test_rejected_attempt_never_runs_full_suite(run, runners: SuiteRunnerFactory) -> None:
    fixer = ScriptedFixer({SUBTRACT: [WRONG_SUBTRACT, WRONG_SUBTRACT]})
    result = run(fixer, [SUBTRACT], max_attempts=2)
    assert result.tests[0].status == OutcomeStatus.UNFIXABLE
    assert runners.runner.run_all_calls == []


def test_baseline_computed_once_per_run(run, runners: SuiteRunnerFactory) -> None:
    fixer = ScriptedFixer({SUBTRACT: [FIX_SUBTRACT], DIV_ZERO: [FIX_DIVIDE]})
    result = run(fixer, [SUBTRACT, DIV_ZERO])
    assert [t.status for t in result.tests] == [OutcomeStatus.FIXED, OutcomeStatus.FIXED]

    calls = runners.runner.run_all_calls
    assert len(calls) == 3  # one baseline + one full run per accepted source fix
    baseline = calls[0]["ops"]
    assert SUBTRACT_BUG in baseline and DIVIDE_BUG in baseline  # pre-attempt tree
    assert SUBTRACT_FIX in calls[1]["ops"] or DIVIDE_FIX in calls[1]["ops"]
    assert SUBTRACT_FIX in calls[2]["ops"] and DIVIDE_FIX in calls[2]["ops"]


def test_baseline_is_taken_on_pre_attempt_tree_and_attempt_restored(
    run, runners: SuiteRunnerFactory
) -> None:
    result = run(ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]}), [SUBTRACT])
    baseline, after = runners.runner.run_all_calls
    assert SUBTRACT_BUG in baseline["ops"] and SUBTRACT_FIX not in baseline["ops"]
    assert SUBTRACT_FIX in after["ops"]  # stash popped before the post-fix full run
    assert "+    return a - b" in result.diff
    assert result.tests[0].files_changed == [OPS_PY]


def test_accepted_full_results_become_new_baseline(run, runners: SuiteRunnerFactory) -> None:
    """EXTRA fails originally, passes after fix 1; fix 2 breaking it again is a regression."""
    runners.suite_rules[EXTRA] = lambda read: EXTRA_MARKER in read(OPS_PY)
    fix_sub = ("edit", OPS_PY, SUBTRACT_BUG, SUBTRACT_FIX + EXTRA_MARKER, "fix subtract")
    fix_div_drop_marker = (
        "edits",
        [(OPS_PY, SUBTRACT_FIX + EXTRA_MARKER, SUBTRACT_FIX), (OPS_PY, DIVIDE_BUG, DIVIDE_FIX)],
        "fix divide (and drop the marker)",
    )
    fixer = ScriptedFixer({SUBTRACT: [fix_sub], DIV_ZERO: [fix_div_drop_marker, FIX_DIVIDE]})
    result = run(fixer, [SUBTRACT, DIV_ZERO])

    sub, div = result.tests
    assert (sub.status, sub.attempts) == (OutcomeStatus.FIXED, 1)
    assert (div.status, div.attempts) == (OutcomeStatus.FIXED, 2)
    (reason,) = _rejections(fixer, DIV_ZERO)
    assert BROKE in reason and EXTRA in reason


def test_full_suite_uses_regression_settings(run, runners: SuiteRunnerFactory) -> None:
    run(
        ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]}),
        [SUBTRACT],
        regression_pytest_args=["-p", "no:cacheprovider"],
        regression_timeout_seconds=77,
    )
    calls = runners.runner.run_all_calls
    assert calls
    assert all(c["extra_args"] == ["-p", "no:cacheprovider"] for c in calls)
    assert all(c["timeout"] == 77 for c in calls)


# ---- pre-existing failures ------------------------------------------------------------------


def test_preexisting_failures_not_counted_and_reported(run, runners: SuiteRunnerFactory) -> None:
    runners.suite_rules[EXTRA] = lambda read: False  # always failing, never requested
    result = run(ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]}), [SUBTRACT])
    (outcome,) = result.tests
    assert (outcome.status, outcome.attempts) == (OutcomeStatus.FIXED, 1)
    # Failing (or ERROR) in the baseline and not requested; skipped tests are not failures.
    assert result.preexisting_failures == sorted([DIV_ZERO, MEAN, FIXTURE_ERROR, EXTRA])
    assert result.warnings == []


def test_preexisting_failures_exclude_all_requested(run) -> None:
    fixer = ScriptedFixer({SUBTRACT: [FIX_SUBTRACT], DIV_ZERO: [FIX_DIVIDE]})
    result = run(fixer, [SUBTRACT, DIV_ZERO])
    assert result.preexisting_failures == sorted([MEAN, FIXTURE_ERROR])


# ---- full suite cannot run ------------------------------------------------------------------


def test_baseline_error_warns_and_skips_regression_checks(
    run, runners: SuiteRunnerFactory, caplog: pytest.LogCaptureFixture
) -> None:
    runners.run_all_errors[1] = TestRunError("pytest timed out after 1800s")
    fixer = ScriptedFixer({SUBTRACT: [BREAK_ADD_AND_FIX_SUBTRACT], DIV_ZERO: [FIX_DIVIDE]})
    result = run(fixer, [SUBTRACT, DIV_ZERO])

    sub, div = result.tests
    assert (sub.status, sub.attempts) == (OutcomeStatus.FIXED, 1)  # accepted normally
    assert div.status == OutcomeStatus.FIXED
    assert len(runners.runner.run_all_calls) == 1  # skipped for the rest of the run
    assert any(w.startswith(SKIPPED_WARNING) for w in result.warnings)
    assert any("timed out" in w for w in result.warnings)
    assert result.preexisting_failures == []
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_baseline_error_still_restores_attempt(run, runners: SuiteRunnerFactory) -> None:
    runners.run_all_errors[1] = TestRunError("boom")
    result = run(ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]}), [SUBTRACT])
    assert result.tests[0].status == OutcomeStatus.FIXED
    assert "+    return a - b" in result.diff


def test_post_fix_full_run_error_warns_and_accepts(run, runners: SuiteRunnerFactory) -> None:
    runners.run_all_errors[2] = TestRunError("pytest crashed\nlong output tail")
    fixer = ScriptedFixer({SUBTRACT: [FIX_SUBTRACT], DIV_ZERO: [FIX_DIVIDE]})
    result = run(fixer, [SUBTRACT, DIV_ZERO])

    sub, div = result.tests
    assert (sub.status, sub.attempts) == (OutcomeStatus.FIXED, 1)  # no attempt burnt
    assert (div.status, div.attempts) == (OutcomeStatus.FIXED, 1)
    assert len(runners.runner.run_all_calls) == 2  # baseline + failed run, then off
    assert result.warnings == [f"{SKIPPED_WARNING} pytest crashed"]
    assert "+    return a - b" in result.diff


def test_flaky_re_run_uses_regression_settings(run, runners: SuiteRunnerFactory) -> None:
    runners.suite_rules[EXTRA] = flaky(lambda read: True, {2: False})
    result = run(
        ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]}),
        [SUBTRACT],
        regression_pytest_args=["-k", "not slow"],
        regression_timeout_seconds=55,
    )
    assert result.tests[0].status == OutcomeStatus.FIXED
    runner = runners.runner
    (rerun,) = [i for i, ids in enumerate(runner.runs) if ids == [EXTRA]]
    assert runner.run_options[rerun] == (["-k", "not slow"], 55)


def test_rejection_reason_separates_failing_and_missing(run, runners: SuiteRunnerFactory) -> None:
    runners.suite_rules[EXTRA] = lambda read: (
        TestStatus.NOT_FOUND if "# HIDE" in read(OPS_PY) else True
    )
    both = (
        "edits",
        [
            (OPS_PY, ADD_BODY, ADD_BODY.replace("a + b\n", "a - b\n")),
            (OPS_PY, SUBTRACT_BUG, SUBTRACT_FIX + "  # HIDE"),
        ],
        "x",
    )
    fixer = ScriptedFixer({SUBTRACT: [both, FIX_SUBTRACT]})
    run(fixer, [SUBTRACT])
    (reason,) = _rejections(fixer, SUBTRACT)
    assert reason == f"{BROKE}: {ERRS_ADD}, {ADD}; no longer collected: {EXTRA}"


def test_suite_file_collection_error_after_fix_is_regression(
    run, runners: SuiteRunnerFactory
) -> None:
    extra_file = EXTRA.split("::", 1)[0]
    runners.suite_rules[EXTRA] = lambda read: (
        TestStatus.NOT_FOUND if "# COLL" in read(OPS_PY) else True
    )
    # The file-level collection error entry only exists while the marker is present.
    runners.suite_rules[extra_file] = lambda read: (
        TestStatus.ERROR if "# COLL" in read(OPS_PY) else TestStatus.NOT_FOUND
    )
    coll = ("edit", OPS_PY, SUBTRACT_BUG, SUBTRACT_FIX + "  # COLL", "x")
    fixer = ScriptedFixer({SUBTRACT: [coll, FIX_SUBTRACT]})
    result = run(fixer, [SUBTRACT])
    assert (result.tests[0].status, result.tests[0].attempts) == (OutcomeStatus.FIXED, 2)
    (reason,) = _rejections(fixer, SUBTRACT)
    assert reason == f"{BROKE}: no longer collected: {EXTRA}"


# ---- restoring the attempt after the baseline run -------------------------------------------

HELPER_PY = "src/calc/helper.py"
FIXER_HELPER = "FIXER = 1\n"


@dataclass
class CreatingFixer(ScriptedFixer):
    """``ScriptedFixer`` that also writes ``creates`` (path -> text) on each DIV_ZERO call."""

    creates: dict[str, str] = field(default_factory=dict)

    def fix(self, request):
        attempt = super().fix(request)
        if request.node_id == DIV_ZERO:
            for rel, text in self.creates.items():
                (Path(request.repo_path) / rel).write_text(text, encoding="utf-8")
            attempt.files_changed = sorted({*attempt.files_changed, *self.creates})
        return attempt


@dataclass
class HookedRunnerFactory(SuiteRunnerFactory):
    """Runs ``on_baseline(repo_path)`` inside the first ``run_all`` call (the baseline)."""

    on_baseline: Any = None

    def __call__(self, repo_path: Path, python: Path, reports_dir: Path, settings: Settings):
        runner = super().__call__(repo_path, python, reports_dir, settings)
        original, hook = runner.run_all, self.on_baseline

        def run_all(extra_args=(), timeout=None):
            if not runner.run_all_calls and hook is not None:
                hook(Path(repo_path))
            return original(extra_args, timeout)

        runner.run_all = run_all  # type: ignore[method-assign]
        return runner


def _two_step_fixer(**kwargs: Any) -> CreatingFixer:
    """SUBTRACT: test-only fix (an accepted checkpoint before the baseline); DIV_ZERO: source."""
    return CreatingFixer({SUBTRACT: [TEST_ONLY_FIX], DIV_ZERO: [FIX_DIVIDE, FIX_DIVIDE]}, **kwargs)


def _hooked(tmp_path: Path, remote: SampleRemote, fixer, hook, **overrides) -> FixResult:
    runners = HookedRunnerFactory(on_baseline=hook)
    runners.rules[SUBTRACT] = _lenient_subtract
    deps = make_deps(tmp_path, fixer, remote=remote, runner_factory=runners, **overrides)
    return fix_failing_tests(REPO_URL, SAMPLE_PR, [SUBTRACT, DIV_ZERO], deps=deps)


def _assert_both_fixes_in_diff(result: FixResult) -> None:
    assert f"+    {EXPECT_8}" in result.diff  # the earlier (test-only) checkpoint survived
    assert '+        raise ValueError("division by zero")' in result.diff


def test_baseline_run_creating_attempts_new_file_does_not_block_restore(
    tmp_path: Path, sample_remote: SampleRemote
) -> None:
    def hook(repo: Path) -> None:
        assert not (repo / HELPER_PY).exists()  # stashed
        (repo / HELPER_PY).write_text("RUN = 1\n", encoding="utf-8")

    fixer = _two_step_fixer(creates={HELPER_PY: FIXER_HELPER})
    result = _hooked(tmp_path, sample_remote, fixer, hook)

    sub, div = result.tests
    assert (sub.status, div.status, div.attempts) == (
        OutcomeStatus.FIXED,
        OutcomeStatus.FIXED,
        1,
    )
    _assert_both_fixes_in_diff(result)
    assert "+FIXER = 1" in result.diff and "RUN = 1" not in result.diff
    assert HELPER_PY in div.files_changed


def test_baseline_run_rewriting_tracked_file_does_not_block_restore(
    tmp_path: Path, sample_remote: SampleRemote
) -> None:
    def hook(repo: Path) -> None:
        ops = repo / OPS_PY
        ops.write_text(ops.read_text(encoding="utf-8") + "# RUN WROTE\n", encoding="utf-8")

    result = _hooked(tmp_path, sample_remote, _two_step_fixer(), hook)
    div = result.tests[1]
    assert (div.status, div.attempts) == (OutcomeStatus.FIXED, 1)
    _assert_both_fixes_in_diff(result)
    assert "RUN WROTE" not in result.diff


def _commits(tmp_path: Path, remote: SampleRemote) -> list[str]:
    repo = run_dir_for(tmp_path) / "repo"
    return git("log", "--format=%s", f"{remote.pr_sha}..HEAD", cwd=repo).splitlines()


def test_failed_unstash_rejects_attempt_and_keeps_checkpoints(
    tmp_path: Path, sample_remote: SampleRemote, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def failing_unstash(self: GitRepo) -> None:
        nonlocal calls
        calls += 1
        raise GitError("simulated: already exists, no checkout")

    monkeypatch.setattr(GitRepo, "unstash", failing_unstash)
    fixer = _two_step_fixer()
    result = _hooked(tmp_path, sample_remote, fixer, None, keep_workspace=True)

    assert calls == 1  # baseline only; attempt 2 reuses the baseline
    div = result.tests[1]
    assert (div.status, div.attempts) == (OutcomeStatus.FIXED, 2)
    (reason,) = _rejections(fixer, DIV_ZERO)
    assert reason.startswith(f"{RESTORE_FAILED_PREFIX} ")
    assert "simulated" in reason
    _assert_both_fixes_in_diff(result)
    repo = run_dir_for(tmp_path) / "repo"
    assert git("stash", "list", cwd=repo) == ""  # the stash was dropped
    assert _commits(tmp_path, sample_remote) == [
        f"ci-fix: fix {DIV_ZERO} (attempt 2)",
        f"ci-fix: fix {SUBTRACT} (attempt 1)",
    ]


def test_restored_attempt_mismatch_rejects_attempt(
    tmp_path: Path, sample_remote: SampleRemote, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = GitRepo.unstash

    def lossy_unstash(self: GitRepo) -> None:
        original(self)
        git("checkout", "--", OPS_PY, cwd=self.path)  # lose part of the attempt

    monkeypatch.setattr(GitRepo, "unstash", lossy_unstash)
    fixer = _two_step_fixer()
    result = _hooked(tmp_path, sample_remote, fixer, None, keep_workspace=True)

    div = result.tests[1]
    assert (div.status, div.attempts) == (OutcomeStatus.FIXED, 2)
    (reason,) = _rejections(fixer, DIV_ZERO)
    assert reason.startswith(f"{RESTORE_FAILED_PREFIX} ")
    _assert_both_fixes_in_diff(result)
    assert len(_commits(tmp_path, sample_remote)) == 2


# ---- end to end on the real pytest ----------------------------------------------------------


def test_e2e_fix_breaking_unrequested_test_is_rejected(
    tmp_path: Path, sample_remote: SampleRemote
) -> None:
    fixer = ScriptedFixer({SUBTRACT: [BREAK_ADD_AND_FIX_SUBTRACT, FIX_SUBTRACT]})
    deps = make_deps(tmp_path, fixer, remote=sample_remote)  # real PytestRunner
    result = fix_failing_tests(REPO_URL, SAMPLE_PR, [SUBTRACT], deps=deps)

    (outcome,) = result.tests
    assert (outcome.status, outcome.attempts) == (OutcomeStatus.FIXED, 2)
    (reason,) = _rejections(fixer, SUBTRACT)
    assert reason.startswith(f"{BROKE}: ")
    assert ADD in reason

    lines = result.diff.splitlines()
    assert [ln for ln in lines if ln.startswith("+") and not ln.startswith("+++")] == [
        "+    return a - b"
    ]
    assert [ln for ln in lines if ln.startswith("-") and not ln.startswith("---")] == [
        f"-    {SUBTRACT_BUG}"
    ]
    assert result.preexisting_failures == sorted(
        [
            "tests/test_ops.py::TestDivide::test_divide_by_zero",
            "tests/test_ops.py::test_mean[xs0-2]",
            "tests/test_ops.py::test_mean[xs1-15]",
            FIXTURE_ERROR,
        ]
    )


# ---- shared test code triggers the regression run ------------------------------------------


@pytest.mark.parametrize(
    ("path", "before", "after", "shared"),
    [
        ("tests/conftest.py", "X = 1\n", "X = 2\n", True),
        ("tests/helpers.py", "def make():\n    return 1\n", "def make():\n    return 2\n", True),
        ("tests/data/cases.json", None, None, True),
        (
            "tests/test_x.py",
            "def test_a():\n    assert 1 == 1\n",
            "def test_a():\n    assert 2 == 2\n",
            False,
        ),
    ],
)
def test_shared_test_files_detected(tmp_path, path, before, after, shared) -> None:
    import subprocess

    from ci_fix.guards.patch_checker import check_patch
    from ci_fix.tools.git import GitRepo

    repo_dir = tmp_path / "r"
    target = repo_dir / path
    target.parent.mkdir(parents=True)
    target.write_text(before if before is not None else '{"a": 1}\n')
    (repo_dir / "tests" / "test_dummy.py").write_text("def test_d():\n    assert True\n")
    for args in (
        ["init", "-q"],
        ["add", "-A"],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "i"],
    ):
        subprocess.run(["git", *args], cwd=repo_dir, check=True)
    target.write_text(after if after is not None else '{"a": 2}\n')
    report = check_patch(GitRepo(repo_dir), "Test change: updated shared helper")
    assert (path in report.shared_test_files_changed) is shared


def test_conftest_only_fix_runs_full_suite(run, runners: SuiteRunnerFactory) -> None:
    """A test-only fix that edits shared test code (conftest.py) triggers the regression run."""
    runners.rules[SUBTRACT] = lambda read: "SHARED_OK = True" in read("tests/conftest.py")
    conftest_fix = (
        "edit",
        "tests/conftest.py",
        "import sys\n",
        "import sys\n\nSHARED_OK = True\n",
        "Root cause: shared setup was missing.\nTest change: conftest needed the flag.",
    )
    result = run(ScriptedFixer({SUBTRACT: [conftest_fix]}), [SUBTRACT])
    (outcome,) = result.tests
    assert outcome.status == OutcomeStatus.FIXED
    assert outcome.source_changed is False
    assert runners.runner.run_all_calls  # baseline + post-fix full run happened
