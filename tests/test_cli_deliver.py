"""CLI tests for slice 8 (delivery): --no-push / --no-comment and the delivery output."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ci_fix import cli
from ci_fix.cli import main
from ci_fix.config import Settings
from ci_fix.logging_setup import LOGGER_NAME
from ci_fix.models import FixResult, OutcomeStatus, TestOutcome

REPO = "https://github.com/octo/repo"
BASE_ARGS = ["--repo", REPO, "--pr", "5", "--tests", "tests/test_a.py::test_x"]
PR_URL = "https://github.com/octo/repo/pull/99"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Run in an empty dir (no stray config.toml) and restore the ci_fix logger afterwards."""
    monkeypatch.chdir(tmp_path)
    for var in ("GITHUB_TOKEN", "ANTHROPIC_API_KEY", "CI_FIX_PUSH", "CI_FIX_COMMENT_ON_PR"):
        monkeypatch.delenv(var, raising=False)
    logger = logging.getLogger(LOGGER_NAME)
    saved = (list(logger.handlers), logger.level, logger.propagate)
    yield
    for h in list(logger.handlers):
        if h not in saved[0]:
            logger.removeHandler(h)
            h.close()
    logger.handlers[:] = saved[0]
    logger.setLevel(saved[1])
    logger.propagate = saved[2]


def _result(status: OutcomeStatus, **fields: Any) -> FixResult:
    test = TestOutcome(requested_name="t0", node_id="tests/test_a.py::t0", status=status)
    return FixResult(
        repo_url=REPO,
        pr_number=5,
        branch="ci-fix/pr-5",
        diff="",
        summary="## ci-fix SUMMARY MARKER",
        tests=[test],
        **fields,
    )


class FakePipeline:
    def __init__(self, result: FixResult) -> None:
        self.result = result
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> FixResult:
        self.calls.append((args, kwargs))
        return self.result

    @property
    def settings(self) -> Settings:
        return self.calls[-1][1]["settings"]


def _install(monkeypatch: pytest.MonkeyPatch, result: FixResult) -> FakePipeline:
    fake = FakePipeline(result)
    monkeypatch.setattr(cli, "fix_failing_tests", fake)
    return fake


# ---- settings -------------------------------------------------------------------------------


def test_settings_defaults() -> None:
    s = Settings()
    assert s.push is True
    assert s.comment_on_pr is True
    assert s.commit_author_name == "ci-fix"
    assert s.commit_author_email == "ci-fix@users.noreply.github.com"


def test_defaults_reach_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install(monkeypatch, _result(OutcomeStatus.FIXED, pr_url=PR_URL, pushed=True))
    assert main(BASE_ARGS) == 0
    assert fake.settings.push is True
    assert fake.settings.comment_on_pr is True


def test_no_push_reaches_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install(monkeypatch, _result(OutcomeStatus.FIXED, pushed=False))
    main([*BASE_ARGS, "--no-push"])
    assert fake.settings.push is False
    assert fake.settings.comment_on_pr is True


def test_no_comment_reaches_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install(monkeypatch, _result(OutcomeStatus.FIXED, pr_url=PR_URL, pushed=True))
    main([*BASE_ARGS, "--no-comment"])
    assert fake.settings.comment_on_pr is False
    assert fake.settings.push is True


def test_both_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install(monkeypatch, _result(OutcomeStatus.FIXED, pushed=False))
    main([*BASE_ARGS, "--no-push", "--no-comment"])
    assert (fake.settings.push, fake.settings.comment_on_pr) == (False, False)


def test_config_file_push_false_is_kept_without_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without --no-push the CLI must not force push back to True."""
    config = tmp_path / "cfg.toml"
    config.write_text("[ci_fix]\npush = false\ncomment_on_pr = false\n", encoding="utf-8")
    fake = _install(monkeypatch, _result(OutcomeStatus.FIXED, pushed=False))
    main([*BASE_ARGS, "--config", str(config)])
    assert (fake.settings.push, fake.settings.comment_on_pr) == (False, False)


# ---- output ---------------------------------------------------------------------------------


def test_prints_pr_url_when_pushed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(monkeypatch, _result(OutcomeStatus.FIXED, pr_url=PR_URL, pushed=True))
    assert main(BASE_ARGS) == 0
    out = capsys.readouterr().out
    assert PR_URL in out
    assert "SUMMARY MARKER" in out


def test_prints_dry_run_when_not_pushed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(
        monkeypatch,
        _result(
            OutcomeStatus.FIXED,
            pushed=False,
            commit_message="ci-fix: fix 1 failing test in #5",
            pr_body="Fixes failing tests in #5 (`feature`).",
        ),
    )
    assert main([*BASE_ARGS, "--no-push"]) == 0
    out = capsys.readouterr().out
    assert "dry run" in out.lower()
    assert "https://github.com/octo/repo/pull/" not in out


def test_prints_no_fixes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(monkeypatch, _result(OutcomeStatus.UNFIXABLE, pushed=False))
    assert main(BASE_ARGS) == 1
    out = capsys.readouterr().out
    assert "no fixes" in out.lower()
    assert "https://github.com/octo/repo/pull/" not in out


def test_help_mentions_new_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "--no-push" in out
    assert "--no-comment" in out
