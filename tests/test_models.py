"""Tests for ci_fix.models (slice 3)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ci_fix.models import (
    FixAttempt,
    FixRequest,
    FixResult,
    NoOpFixer,
    OutcomeStatus,
    TestOutcome,
)


def test_outcome_status_values() -> None:
    assert {s.value for s in OutcomeStatus} == {
        "fixed",
        "already_passing",
        "unfixable",
        "not_found",
        "ambiguous",
    }
    assert OutcomeStatus("fixed") is OutcomeStatus.FIXED


def test_test_outcome_defaults() -> None:
    o = TestOutcome(requested_name="t", node_id="a.py::t", status=OutcomeStatus.FIXED)
    assert o.reason == ""
    assert o.attempts == 0
    assert o.files_changed == []


def test_test_outcome_files_changed_not_shared() -> None:
    a = TestOutcome(requested_name="a", node_id=None, status=OutcomeStatus.NOT_FOUND)
    b = TestOutcome(requested_name="b", node_id=None, status=OutcomeStatus.NOT_FOUND)
    a.files_changed.append("x.py")
    assert b.files_changed == []


def test_fix_request_defaults(tmp_path: Path) -> None:
    r = FixRequest(
        node_id="a.py::t",
        repo_path=tmp_path,
        attempt=1,
        max_attempts=3,
        failure_message="assert 1 == 2",
        failure_details="tb",
    )
    assert r.previous_attempts == []


@pytest.mark.parametrize("outcome", ["changed", "unfixable", "no_change"])
def test_fix_attempt_outcomes(outcome: str) -> None:
    a = FixAttempt(node_id="a.py::t", attempt=1, outcome=outcome)
    assert a.explanation == ""
    assert a.files_changed == []


def test_fix_attempt_rejects_unknown_outcome() -> None:
    with pytest.raises(ValidationError):
        FixAttempt(node_id="a.py::t", attempt=1, outcome="maybe")


def test_fix_result_fixed_and_unfixable_properties() -> None:
    tests = [
        TestOutcome(requested_name="a", node_id="x::a", status=OutcomeStatus.FIXED, attempts=1),
        TestOutcome(requested_name="b", node_id="x::b", status=OutcomeStatus.UNFIXABLE),
        TestOutcome(requested_name="c", node_id="x::c", status=OutcomeStatus.ALREADY_PASSING),
        TestOutcome(requested_name="d", node_id=None, status=OutcomeStatus.NOT_FOUND),
        TestOutcome(requested_name="e", node_id="x::e", status=OutcomeStatus.FIXED),
    ]
    result = FixResult(
        repo_url="https://github.com/o/r",
        pr_number=3,
        branch="ci-fix/pr-3",
        diff="",
        summary="s",
        tests=tests,
    )
    assert result.pr_url is None
    assert [t.requested_name for t in result.fixed] == ["a", "e"]
    assert [t.requested_name for t in result.unfixable] == ["b"]
    assert [t.requested_name for t in result.tests] == ["a", "b", "c", "d", "e"]


def test_noop_fixer_reports_unfixable(tmp_path: Path) -> None:
    req = FixRequest(
        node_id="a.py::t",
        repo_path=tmp_path,
        attempt=2,
        max_attempts=3,
        failure_message="m",
        failure_details="d",
    )
    attempt = NoOpFixer().fix(req)
    assert attempt.outcome == "unfixable"
    assert attempt.node_id == "a.py::t"
    assert attempt.attempt == 2
    assert "No fixer configured" in attempt.explanation
    assert attempt.files_changed == []


def test_public_exports() -> None:
    import ci_fix

    for name in (
        "fix_failing_tests",
        "FixResult",
        "TestOutcome",
        "OutcomeStatus",
        "FixRequest",
        "FixAttempt",
        "Fixer",
        "NoOpFixer",
    ):
        assert hasattr(ci_fix, name), name


def test_fix_attempt_verification_defaults() -> None:
    a = FixAttempt(node_id="a.py::t", attempt=1, outcome="changed")
    assert a.accepted is None
    assert a.rejection_reason == ""
