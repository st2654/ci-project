"""Pipeline tests for the integrity guards (slice 5): the patch checker and the optional
test-change reviewer, on the sample remote with a ``FakeRunner`` and scripted fixers.

The runner rules here are deliberately lenient (a cheat makes the target "pass"), so every
rejection below comes from the integrity guards, not from verification.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pytest
from pipeline_helpers import (
    ERRS,
    FIX_SUBTRACT,
    FIXTURE_ERROR,
    MEAN,
    MEAN_BUG,
    OPS_PY,
    REPO_URL,
    SAMPLE_PR,
    SAMPLE_RULES,
    SUBTRACT,
    SUBTRACT_FIX,
    TEST_OPS_PY,
    FakeRunnerFactory,
    Read,
    SampleRemote,
    ScriptedFixer,
    make_deps,
    run_dirs_for,
)

from ci_fix.guards.reviewer import ReviewVerdict
from ci_fix.models import FixerFatalError, FixRequest, FixResult, OutcomeStatus, TestOutcome
from ci_fix.pipeline import fix_failing_tests
from ci_fix.tools.pytest_runner import TestStatus

INTEGRITY = "integrity check failed:"
REVIEW_REJECTED = "reviewer rejected test change"

# ---- scripted actions -----------------------------------------------------------------------

SKIP_CHEAT = (
    "edit",
    TEST_OPS_PY,
    "def test_subtract():\n",
    'def test_subtract():\n    pytest.skip("not today")\n',
    "skip it for now",
)
EXPECT_8 = "assert subtract(5, 3) == 8"
JUSTIFIED_EXPECTATION = (
    "edit",
    TEST_OPS_PY,
    "assert subtract(5, 3) == 2",
    EXPECT_8,
    "Root cause: the test expected the wrong value.\nTest change: subtract is documented to add.",
)
UNJUSTIFIED_EXPECTATION = (
    "edit",
    TEST_OPS_PY,
    "assert subtract(5, 3) == 2",
    EXPECT_8,
    "Made the test match the output.",
)
SWALLOW_MEAN = (
    "edit",
    OPS_PY,
    MEAN_BUG,
    "try:\n        return sum(xs) / len(xs)\n    except Exception:\n        return None",
    "guard the division",
)
FIXTURE_CHEAT = (
    "edits",
    [
        (ERRS, "@pytest.fixture\ndef broken_resource():", "def broken_resource():"),
        (
            ERRS,
            "def test_fixture_error(broken_resource):\n    assert broken_resource is not None\n",
            "def test_fixture_error():\n"
            '    with pytest.raises(RuntimeError, match="fixture setup failed"):\n'
            "        broken_resource()\n",
        ),
    ],
    "Test change: the fixture is supposed to raise, so assert that it does.",
)


# ---- lenient runner rules -------------------------------------------------------------------


def _lenient_subtract(read: Read) -> bool | TestStatus:
    tests = read(TEST_OPS_PY)
    if "def test_subtract" not in tests:
        return TestStatus.NOT_FOUND
    return SUBTRACT_FIX in read(OPS_PY) or "pytest.skip(" in tests or EXPECT_8 in tests


def _lenient_fixture_error(read: Read) -> bool | TestStatus:
    return "pytest.raises(RuntimeError" in read(ERRS) or TestStatus.ERROR


LENIENT_RULES = {
    **SAMPLE_RULES,
    SUBTRACT: _lenient_subtract,
    FIXTURE_ERROR: _lenient_fixture_error,
}


# ---- fake reviewers -------------------------------------------------------------------------


@dataclasses.dataclass
class FakeReviewer:
    approved: bool = True
    reason: str = "looks fine"
    calls: list[tuple[FixRequest, str, str]] = dataclasses.field(default_factory=list)

    def review(self, request: FixRequest, diff: str, explanation: str) -> ReviewVerdict:
        self.calls.append((request, diff, explanation))
        return ReviewVerdict(approved=self.approved, reason=self.reason)


class FatalReviewer:
    calls = 0

    def review(self, request: FixRequest, diff: str, explanation: str) -> ReviewVerdict:
        FatalReviewer.calls += 1
        raise FixerFatalError("Claude API rejected the request: credit balance is too low")


# ---- harness --------------------------------------------------------------------------------


@pytest.fixture
def runners() -> FakeRunnerFactory:
    return FakeRunnerFactory(rules=dict(LENIENT_RULES))


@pytest.fixture
def run(tmp_path: Path, sample_remote: SampleRemote, runners: FakeRunnerFactory):
    def _run(fixer, tests: list[str], reviewer=None, **overrides) -> FixResult:
        deps = make_deps(tmp_path, fixer, remote=sample_remote, runner_factory=runners, **overrides)
        assert deps.reviewer is None  # default
        if reviewer is not None:
            deps = dataclasses.replace(deps, reviewer=reviewer)
        return fix_failing_tests(REPO_URL, SAMPLE_PR, tests, deps=deps)

    return _run


def _only(result: FixResult) -> TestOutcome:
    assert len(result.tests) == 1
    return result.tests[0]


# ---- patch checker in the loop --------------------------------------------------------------


def test_cheat_is_rejected_rolled_back_and_reason_fed_to_next_attempt(
    run, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="ci_fix")
    fixer = ScriptedFixer({SUBTRACT: [SKIP_CHEAT, FIX_SUBTRACT]})
    result = run(fixer, [SUBTRACT])

    outcome = _only(result)
    assert (outcome.status, outcome.attempts) == (OutcomeStatus.FIXED, 2)
    assert "pytest.skip" not in result.diff  # the cheat was rolled back
    assert "+    return a - b" in result.diff
    assert outcome.source_changed is True

    first, second = fixer.requests
    (prev,) = second.previous_attempts
    assert prev.accepted is False
    assert prev.rejection_reason.startswith(INTEGRITY)
    assert "skip_added" in prev.rejection_reason
    assert second.failure_message == first.failure_message

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("[integrity]" in r.getMessage() for r in warnings)


def test_every_attempt_cheating_is_unfixable_with_integrity_reason(run) -> None:
    fixer = ScriptedFixer({SUBTRACT: [SKIP_CHEAT] * 3})
    result = run(fixer, [SUBTRACT])

    outcome = _only(result)
    assert (outcome.status, outcome.attempts) == (OutcomeStatus.UNFIXABLE, 3)
    assert INTEGRITY in outcome.reason
    assert "skip_added" in outcome.reason
    assert result.diff == ""
    assert len(fixer.requests) == 3


def test_unjustified_expectation_change_is_rejected(run) -> None:
    fixer = ScriptedFixer({SUBTRACT: [UNJUSTIFIED_EXPECTATION]})
    result = run(fixer, [SUBTRACT], max_attempts=1)

    outcome = _only(result)
    assert outcome.status == OutcomeStatus.UNFIXABLE
    assert INTEGRITY in outcome.reason
    assert "unjustified_test_change" in outcome.reason
    assert result.diff == ""


def test_swallowing_error_in_source_is_rejected(run) -> None:
    fixer = ScriptedFixer({MEAN: [SWALLOW_MEAN]})
    result = run(fixer, [MEAN], max_attempts=1)

    outcome = _only(result)
    assert outcome.status == OutcomeStatus.UNFIXABLE
    assert INTEGRITY in outcome.reason
    assert "error_swallowed" in outcome.reason
    assert "except Exception" not in result.diff


def test_production_fixture_cheat_is_rejected(run) -> None:
    fixer = ScriptedFixer({FIXTURE_ERROR: [FIXTURE_CHEAT] * 3})
    result = run(fixer, [FIXTURE_ERROR])

    outcome = _only(result)
    assert outcome.status == OutcomeStatus.UNFIXABLE
    assert INTEGRITY in outcome.reason
    assert "expects_exception_added" in outcome.reason
    assert "fixture_removed" in outcome.reason
    assert result.diff == ""


def test_justified_expectation_change_is_fixed_and_reported(run) -> None:
    fixer = ScriptedFixer({SUBTRACT: [JUSTIFIED_EXPECTATION]})
    result = run(fixer, [SUBTRACT])

    outcome = _only(result)
    assert (outcome.status, outcome.attempts) == (OutcomeStatus.FIXED, 1)
    assert outcome.source_changed is False
    assert outcome.test_changes
    assert all(isinstance(change, str) for change in outcome.test_changes)
    assert any("8" in change for change in outcome.test_changes)
    assert "Test change: subtract is documented to add." in outcome.explanation
    assert f"+    {EXPECT_8}" in result.diff


def test_source_fix_sets_source_changed_and_no_test_changes(run) -> None:
    fixer = ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]})
    result = run(fixer, [SUBTRACT])

    outcome = _only(result)
    assert outcome.status == OutcomeStatus.FIXED
    assert outcome.source_changed is True
    assert outcome.test_changes == []
    assert outcome.explanation == FIX_SUBTRACT[-1]


# ---- reviewer -------------------------------------------------------------------------------


def test_reviewer_rejecting_test_change_rejects_attempt(run) -> None:
    reviewer = FakeReviewer(approved=False, reason="the test was right, fix the source")
    fixer = ScriptedFixer({SUBTRACT: [JUSTIFIED_EXPECTATION, FIX_SUBTRACT]})
    result = run(fixer, [SUBTRACT], reviewer=reviewer, review_test_changes=True)

    outcome = _only(result)
    assert (outcome.status, outcome.attempts) == (OutcomeStatus.FIXED, 2)
    assert EXPECT_8 not in result.diff  # the rejected test change was rolled back
    assert outcome.source_changed is True

    (prev,) = fixer.requests[1].previous_attempts
    assert prev.accepted is False
    assert prev.rejection_reason.startswith(REVIEW_REJECTED)
    assert "the test was right, fix the source" in prev.rejection_reason

    # Called once: for the test change only, not for the source-only second attempt.
    (call,) = reviewer.calls
    request, diff, explanation = call
    assert request.node_id == SUBTRACT
    assert EXPECT_8 in diff
    assert "Test change:" in explanation


def test_reviewer_approving_test_change_accepts_attempt(run) -> None:
    reviewer = FakeReviewer(approved=True)
    fixer = ScriptedFixer({SUBTRACT: [JUSTIFIED_EXPECTATION]})
    result = run(fixer, [SUBTRACT], reviewer=reviewer, review_test_changes=True)

    outcome = _only(result)
    assert (outcome.status, outcome.attempts) == (OutcomeStatus.FIXED, 1)
    assert outcome.test_changes
    assert len(reviewer.calls) == 1


def test_reviewer_rejecting_every_attempt_is_unfixable(run) -> None:
    reviewer = FakeReviewer(approved=False, reason="not convinced")
    fixer = ScriptedFixer({SUBTRACT: [JUSTIFIED_EXPECTATION] * 2})
    result = run(fixer, [SUBTRACT], reviewer=reviewer, review_test_changes=True, max_attempts=2)

    outcome = _only(result)
    assert outcome.status == OutcomeStatus.UNFIXABLE
    assert REVIEW_REJECTED in outcome.reason
    assert result.diff == ""
    assert len(reviewer.calls) == 2


def test_reviewer_not_called_for_source_only_fix(run) -> None:
    reviewer = FakeReviewer(approved=False)
    fixer = ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]})
    result = run(fixer, [SUBTRACT], reviewer=reviewer, review_test_changes=True)

    assert _only(result).status == OutcomeStatus.FIXED
    assert reviewer.calls == []


def test_reviewer_not_called_when_checker_rejects(run) -> None:
    reviewer = FakeReviewer(approved=True)
    fixer = ScriptedFixer({SUBTRACT: [SKIP_CHEAT]})
    result = run(fixer, [SUBTRACT], reviewer=reviewer, review_test_changes=True, max_attempts=1)

    assert INTEGRITY in _only(result).reason
    assert reviewer.calls == []


def test_no_reviewer_accepts_justified_test_change(run) -> None:
    fixer = ScriptedFixer({SUBTRACT: [JUSTIFIED_EXPECTATION]})
    result = run(fixer, [SUBTRACT])  # deps.reviewer is None
    assert _only(result).status == OutcomeStatus.FIXED


def test_fatal_reviewer_error_stops_run(run, tmp_path: Path) -> None:
    FatalReviewer.calls = 0
    fixer = ScriptedFixer({SUBTRACT: [JUSTIFIED_EXPECTATION, FIX_SUBTRACT]})
    with pytest.raises(FixerFatalError, match="credit balance"):
        run(fixer, [SUBTRACT, MEAN], reviewer=FatalReviewer(), review_test_changes=True)
    assert FatalReviewer.calls == 1
    assert len(fixer.requests) == 1  # no further attempts, no other tests tried
    assert run_dirs_for(tmp_path) == []  # workspace still cleaned up


# ---- checker crash / context / reviewer retries ----------------------------------------------


def test_checker_crash_rejects_attempt(run, monkeypatch: pytest.MonkeyPatch) -> None:
    import ci_fix.graph as graph_module

    def boom(*args, **kwargs):
        raise RuntimeError("checker bug")

    monkeypatch.setattr(graph_module, "check_patch", boom)
    result = run(ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]}), [SUBTRACT], max_attempts=1)
    outcome = _only(result)
    assert outcome.status == OutcomeStatus.UNFIXABLE
    assert f"{INTEGRITY} checker error: RuntimeError: checker bug" in outcome.reason
    assert result.diff == ""  # rolled back


def test_special_casing_the_target_test_inputs_is_rejected(run) -> None:
    special = (
        "edit",
        OPS_PY,
        "return a + b  # SEEDED BUG: should be a - b",
        "if a == 5 and b == 3:\n        return 2\n    return a + b",
        "handle the case",
    )
    result = run(ScriptedFixer({SUBTRACT: [special]}), [SUBTRACT], max_attempts=1)
    outcome = _only(result)
    assert outcome.status == OutcomeStatus.UNFIXABLE
    assert "[special_case_inputs]" in outcome.reason


@dataclasses.dataclass
class FlakyReviewer:
    failures: int
    calls: int = 0

    def review(self, request: FixRequest, diff: str, explanation: str) -> ReviewVerdict:
        self.calls += 1
        if self.calls <= self.failures:
            raise TimeoutError("reviewer timed out")
        return ReviewVerdict(approved=True, reason="ok")


def test_reviewer_error_is_retried_once(run) -> None:
    reviewer = FlakyReviewer(failures=1)
    fixer = ScriptedFixer({SUBTRACT: [JUSTIFIED_EXPECTATION]})
    result = run(fixer, [SUBTRACT], reviewer=reviewer, review_test_changes=True)
    outcome = _only(result)
    assert (outcome.status, outcome.attempts) == (OutcomeStatus.FIXED, 1)
    assert reviewer.calls == 2


def test_reviewer_failing_twice_rejects_as_unavailable(run) -> None:
    reviewer = FlakyReviewer(failures=99)
    fixer = ScriptedFixer({SUBTRACT: [JUSTIFIED_EXPECTATION, FIX_SUBTRACT]})
    result = run(fixer, [SUBTRACT], reviewer=reviewer, review_test_changes=True)
    outcome = _only(result)
    assert (outcome.status, outcome.attempts) == (OutcomeStatus.FIXED, 2)
    assert reviewer.calls == 2  # two tries for the test change; none for the source fix
    (prev,) = fixer.requests[1].previous_attempts
    assert prev.rejection_reason == "reviewer unavailable: TimeoutError: reviewer timed out"
