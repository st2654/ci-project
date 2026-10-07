"""Result and fixer models shared by the pipeline, the graph and fixer implementations."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ci_fix.tools.pytest_runner import TestResult


class OutcomeStatus(StrEnum):
    """Final status of one requested test."""

    FIXED = "fixed"
    ALREADY_PASSING = "already_passing"
    UNFIXABLE = "unfixable"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"


class TestOutcome(BaseModel):
    """What happened to one test name the user passed in."""

    __test__ = False

    requested_name: str
    node_id: str | None
    status: OutcomeStatus
    reason: str = ""
    attempts: int = 0
    files_changed: list[str] = Field(default_factory=list)


class FixAttempt(BaseModel):
    """The result of one fixer call for one test."""

    node_id: str
    attempt: int
    outcome: Literal["changed", "unfixable", "no_change"]
    explanation: str = ""
    files_changed: list[str] = Field(default_factory=list)
    # Set by the pipeline after verification: None = not verified (e.g. fixer gave up).
    accepted: bool | None = None
    rejection_reason: str = ""


class FixRequest(BaseModel):
    """Everything a fixer needs to attempt a fix for one failing test."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_id: str
    repo_path: Path
    attempt: int  # 1-based
    max_attempts: int
    failure_message: str
    failure_details: str
    previous_attempts: list[FixAttempt] = Field(default_factory=list)
    pr_title: str = ""
    pr_body: str = ""
    # Diff of what the PR changed (merge-base..PR head), truncated; "" if unknown.
    pr_diff: str = ""
    # Other requested tests that are still failing (they may share the root cause).
    other_failing_tests: list[str] = Field(default_factory=list)
    # Runs the target test (only) and returns its result; set by the pipeline.
    run_test: Callable[[str], TestResult] | None = Field(default=None, exclude=True, repr=False)


class Fixer(Protocol):
    """Edits files under ``request.repo_path`` to make ``request.node_id`` pass."""

    def fix(self, request: FixRequest) -> FixAttempt: ...


class FixerFatalError(Exception):
    """A fixer error no retry can fix (bad API key, no credit, unsupported parameter).

    The pipeline stops the whole run instead of counting it as a failed attempt.
    """


class NoOpFixer:
    """Default fixer until a real one exists: changes nothing and reports unfixable."""

    def fix(self, request: FixRequest) -> FixAttempt:
        return FixAttempt(
            node_id=request.node_id,
            attempt=request.attempt,
            outcome="unfixable",
            explanation="No fixer configured",
        )


class FixResult(BaseModel):
    """Return value of :func:`ci_fix.fix_failing_tests`."""

    repo_url: str
    pr_number: int
    branch: str | None
    diff: str
    summary: str
    tests: list[TestOutcome]  # in the order the user passed the names
    pr_url: str | None = None

    @property
    def fixed(self) -> list[TestOutcome]:
        return [t for t in self.tests if t.status == OutcomeStatus.FIXED]

    @property
    def unfixable(self) -> list[TestOutcome]:
        return [t for t in self.tests if t.status == OutcomeStatus.UNFIXABLE]
