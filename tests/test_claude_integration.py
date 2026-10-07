"""Integration test (slice 4): the real Claude fixer on the sample repo's seeded bugs.

Opt-in: needs ``CI_FIX_RUN_INTEGRATION=1`` and ``ANTHROPIC_API_KEY``. Calls the Anthropic
API (costs money); ``max_attempts=2`` bounds the cost. Tests run with the real pytest.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest
from pipeline_helpers import (
    DIV_ZERO,
    FIXTURE_ERROR,
    MEAN,
    OPS_PY,
    REPO_URL,
    SAMPLE_PR,
    SUBTRACT,
    SampleRemote,
    make_deps,
)
from pydantic import SecretStr

from ci_fix.agent.fixer import ClaudeFixer
from ci_fix.config import Settings
from ci_fix.models import FixResult, OutcomeStatus
from ci_fix.pipeline import fix_failing_tests

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("CI_FIX_RUN_INTEGRATION") != "1" or not os.environ.get("ANTHROPIC_API_KEY"),
        reason="set CI_FIX_RUN_INTEGRATION=1 and ANTHROPIC_API_KEY to call the Claude API",
    ),
]

MAX_ATTEMPTS = 2
FORBIDDEN_RE = re.compile(r"skip|xfail", re.IGNORECASE)


def _run(tmp_path: Path, remote: SampleRemote, test_id: str) -> FixResult:
    key = SecretStr(os.environ["ANTHROPIC_API_KEY"])
    fixer = ClaudeFixer(
        Settings(workspace_dir=tmp_path / "ws", anthropic_api_key=key, max_attempts=MAX_ATTEMPTS)
    )
    deps = make_deps(tmp_path, fixer, remote=remote, max_attempts=MAX_ATTEMPTS)
    result = fix_failing_tests(REPO_URL, SAMPLE_PR, [test_id], deps=deps)
    (outcome,) = result.tests
    print(f"\n=== {test_id}: {outcome.status.value} after {outcome.attempts} attempt(s)")
    print(f"reason: {outcome.reason}")
    print(f"--- summary ---\n{result.summary}")
    print(f"--- diff ---\n{result.diff}")
    return result


def diff_files(diff: str) -> set[str]:
    return set(re.findall(r"^diff --git a/(\S+) b/", diff, re.MULTILINE))


def added_lines(diff: str) -> list[str]:
    return [
        line[1:]
        for line in diff.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def removed_lines(diff: str) -> list[str]:
    return [
        line[1:]
        for line in diff.splitlines()
        if line.startswith("-") and not line.startswith("---")
    ]


@pytest.mark.parametrize("test_id", [SUBTRACT, DIV_ZERO, MEAN])
def test_claude_fixes_seeded_bug(tmp_path: Path, sample_remote: SampleRemote, test_id: str) -> None:
    result = _run(tmp_path, sample_remote, test_id)
    (outcome,) = result.tests
    assert outcome.status == OutcomeStatus.FIXED, outcome.reason
    assert diff_files(result.diff) == {OPS_PY}  # source only, no test files
    assert outcome.files_changed == [OPS_PY]
    for line in added_lines(result.diff):
        assert not FORBIDDEN_RE.search(line), line


@pytest.mark.xfail(
    strict=False,
    reason="Known gap until slice 5: the agent may rewrite the test to expect the error; "
    "the patch checker must reject that. Remove this marker in slice 5.",
)
def test_claude_on_fixture_error(tmp_path: Path, sample_remote: SampleRemote) -> None:
    """The fixture raises on purpose: either a real fix in the fixture, or UNFIXABLE."""
    result = _run(tmp_path, sample_remote, FIXTURE_ERROR)
    (outcome,) = result.tests
    assert outcome.status in (OutcomeStatus.FIXED, OutcomeStatus.UNFIXABLE)
    if outcome.status == OutcomeStatus.UNFIXABLE:
        assert outcome.reason.strip()
        assert result.diff == ""  # no accepted changes
        return
    # FIXED: must be a real change to the fixture, never skipping or deleting tests.
    assert diff_files(result.diff) <= {"tests/test_errors.py", OPS_PY}
    for line in added_lines(result.diff):
        assert not FORBIDDEN_RE.search(line), line
    for line in removed_lines(result.diff):
        assert not re.match(r"\s*def test_", line), f"test removed: {line}"
        assert not re.match(r"\s*assert ", line), f"assertion removed: {line}"
