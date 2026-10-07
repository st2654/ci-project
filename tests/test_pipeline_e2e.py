"""End-to-end pipeline tests on the real pytest runner and the sample repo.

Slower than ``test_pipeline.py`` (each run starts pytest several times), so only the key
paths are covered here; graph logic is tested on the ``FakeRunner``.
"""

from __future__ import annotations

from pathlib import Path

from conftest import git
from pipeline_helpers import (
    DELETE_SUBTRACT_TEST,
    FIX_SUBTRACT,
    OPS_PY,
    REPO_URL,
    SAMPLE_PR,
    SUBTRACT,
    WRONG_SUBTRACT,
    SampleRemote,
    ScriptedFixer,
    make_deps,
    run_dir_for,
    run_dirs_for,
)

from ci_fix.models import FixResult, OutcomeStatus
from ci_fix.pipeline import fix_failing_tests


def _run(tmp_path: Path, remote: SampleRemote, fixer, tests: list[str], **overrides) -> FixResult:
    deps = make_deps(tmp_path, fixer, remote=remote, **overrides)
    return fix_failing_tests(REPO_URL, SAMPLE_PR, tests, deps=deps)


def test_fixed_in_one_attempt(tmp_path: Path, sample_remote: SampleRemote) -> None:
    fixer = ScriptedFixer({SUBTRACT: [FIX_SUBTRACT]})
    result = _run(tmp_path, sample_remote, fixer, [SUBTRACT])

    (outcome,) = result.tests
    assert (outcome.status, outcome.attempts, outcome.files_changed) == (
        OutcomeStatus.FIXED,
        1,
        [OPS_PY],
    )
    assert "-    return a + b  # SEEDED BUG" in result.diff
    assert "+    return a - b" in result.diff
    (req,) = fixer.requests
    assert "assert 8 == 2" in req.failure_message + req.failure_details
    assert run_dirs_for(tmp_path) == []  # workspace cleaned up


def test_second_attempt_after_wrong_edit(tmp_path: Path, sample_remote: SampleRemote) -> None:
    fixer = ScriptedFixer({SUBTRACT: [WRONG_SUBTRACT, FIX_SUBTRACT]})
    result = _run(tmp_path, sample_remote, fixer, [SUBTRACT], keep_workspace=True)

    (outcome,) = result.tests
    assert (outcome.status, outcome.attempts) == (OutcomeStatus.FIXED, 2)
    assert "a * b" not in result.diff
    assert "+    return a - b" in result.diff
    (prev,) = fixer.requests[1].previous_attempts
    assert prev.accepted is False
    assert "15" in prev.rejection_reason  # subtract(5, 3) returned 5 * 3

    repo = run_dir_for(tmp_path) / "repo"
    log = git("log", "--format=%s", f"{sample_remote.pr_sha}..refs/ci-fix/checkpoints", cwd=repo)
    assert log.splitlines() == [f"ci-fix: fix {SUBTRACT} (attempt 2)"]


def test_fix_that_deletes_the_test_is_rejected(tmp_path: Path, sample_remote: SampleRemote) -> None:
    fixer = ScriptedFixer({SUBTRACT: [DELETE_SUBTRACT_TEST]})
    result = _run(tmp_path, sample_remote, fixer, [SUBTRACT], max_attempts=1)

    (outcome,) = result.tests
    assert outcome.status == OutcomeStatus.UNFIXABLE
    # Slice 5: the patch checker rejects the deletion before verification runs.
    assert outcome.reason.startswith("still failing after 1 attempt(s): integrity check failed:")
    assert "[test_removed]" in outcome.reason
    assert result.diff == ""
