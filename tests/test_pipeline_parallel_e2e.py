"""Slice 7 end to end: parallel fixing on the real pytest runner.

The sample repo plus a second module (``strings.py``) gives two independent seeded bugs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from conftest import git
from parallel_helpers import (
    FIX_SHOUT,
    SHOUT,
    SHOUT_FIX,
    ConcurrentFixer,
    make_two_module_remote,
    overlapped,
)
from pipeline_helpers import (
    FIX_SUBTRACT,
    REPO_URL,
    SAMPLE_PR,
    SUBTRACT,
    SUBTRACT_FIX,
    make_deps,
    run_dir_for,
)

from ci_fix.models import FixAttempt, FixRequest, OutcomeStatus
from ci_fix.pipeline import fix_failing_tests
from ci_fix.tools.pytest_runner import TestStatus


class _CheckingFixer(ConcurrentFixer):
    """Applies the scripted edit, then runs the target via ``request.run_test``."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.checks: dict[str, tuple[Path, TestStatus]] = {}

    def fix(self, request: FixRequest) -> FixAttempt:
        attempt = super().fix(request)
        status = request.run_test(request.node_id).status
        with self._lock:
            self.checks[request.node_id] = (Path(request.repo_path), status)
        return attempt


def test_independent_bugs_fixed_in_parallel_with_real_pytest(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="ci_fix")
    remote = make_two_module_remote(tmp_path / "remote")
    fixer = _CheckingFixer({SUBTRACT: [FIX_SUBTRACT], SHOUT: [FIX_SHOUT]}, delay=0.5)
    deps = make_deps(tmp_path, fixer, remote=remote, keep_workspace=True)

    result = fix_failing_tests(REPO_URL, SAMPLE_PR, [SUBTRACT, SHOUT], deps=deps)

    outcomes = {t.node_id: t for t in result.tests}
    assert (outcomes[SUBTRACT].status, outcomes[SUBTRACT].attempts) == (OutcomeStatus.FIXED, 1)
    assert (outcomes[SHOUT].status, outcomes[SHOUT].attempts) == (OutcomeStatus.FIXED, 1)
    assert f"+    {SUBTRACT_FIX}" in result.diff
    assert f"+    {SHOUT_FIX}" in result.diff

    first, second = fixer.calls
    assert overlapped(first, second)

    # The fix made in each worktree passes that worktree's own run_test: the worktree's
    # code is what gets imported, not the main checkout's.
    run_dir = run_dir_for(tmp_path)
    main = run_dir / "repo"
    for nid in (SUBTRACT, SHOUT):
        repo_path, status = fixer.checks[nid]
        assert (run_dir / "worktrees").resolve() in repo_path.resolve().parents
        assert status == TestStatus.PASSED

    log = git(
        "log", "--format=%s", "--reverse", f"{remote.pr_sha}..refs/ci-fix/checkpoints", cwd=main
    )
    assert log.splitlines() == [
        f"ci-fix: fix {SUBTRACT} (attempt 1)",
        f"ci-fix: fix {SHOUT} (attempt 1)",
    ]
    assert git("worktree", "list", "--porcelain", cwd=main).count("worktree ") == 1
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "[parallel] round 1: fixing 2 test(s) in parallel" in text
