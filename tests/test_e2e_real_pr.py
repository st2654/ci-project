"""Slice 9: end-to-end on a real PR (dry run — nothing is written to GitHub).

Runs only with CI_FIX_RUN_INTEGRATION=1 and both ANTHROPIC_API_KEY and GITHUB_TOKEN set.
Target: st2654/python-arithmetic-tests#3, which has one failure of each kind.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ci_fix import OutcomeStatus, fix_failing_tests, load_settings

REPO = "https://github.com/st2654/python-arithmetic-tests"
PR = 3

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("CI_FIX_RUN_INTEGRATION") != "1"
        or not os.environ.get("ANTHROPIC_API_KEY")
        or not os.environ.get("GITHUB_TOKEN"),
        reason="needs CI_FIX_RUN_INTEGRATION=1, ANTHROPIC_API_KEY and GITHUB_TOKEN",
    ),
]


def test_mixed_failures_end_to_end(tmp_path: Path) -> None:
    settings = load_settings(path=None, env=os.environ).model_copy(
        update={"workspace_dir": tmp_path / "ws", "push": False, "review_test_changes": True}
    )
    result = fix_failing_tests(
        REPO,
        PR,
        ["test_area_rectangle", "test_perimeter_rectangle", "test_slugify", "test_fetch_status"],
        settings=settings,
    )
    by_name = {t.requested_name: t for t in result.tests}

    area, perimeter = by_name["test_area_rectangle"], by_name["test_perimeter_rectangle"]
    slug, remote = by_name["test_slugify"], by_name["test_fetch_status"]
    assert area.status == OutcomeStatus.FIXED and area.source_changed
    assert slug.status == OutcomeStatus.FIXED and slug.source_changed
    assert perimeter.status == OutcomeStatus.FIXED and perimeter.test_changes  # wrong test, flagged
    assert remote.status == OutcomeStatus.UNFIXABLE and remote.reason  # needs an internal service

    assert result.pushed is False and result.pr_url is None  # dry run
    assert "test_remote.py" not in result.diff  # nothing faked for the unfixable test
    for marker in ("skip", "xfail", "monkeypatch", "mock"):
        assert (
            marker
            not in "".join(ln for ln in result.diff.splitlines() if ln.startswith("+")).lower()
        )
    assert "### Test changes" in result.pr_body and "### Unfixable" in result.pr_body
    assert len(result.pr_body) < 4000  # readable in 2–3 minutes
    assert not (tmp_path / "ws").exists() or not any((tmp_path / "ws").iterdir())
