"""Tests for ci_fix.cli (slice 3: runs the pipeline; fix_failing_tests is faked)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ci_fix import cli
from ci_fix.cli import main
from ci_fix.config import ConfigError, Settings
from ci_fix.logging_setup import LOGGER_NAME
from ci_fix.models import FixResult, OutcomeStatus, TestOutcome
from ci_fix.tools.git import GitError
from ci_fix.tools.github import GitHubError
from ci_fix.tools.pytest_runner import TestRunError
from ci_fix.tools.test_env import TestEnvError

REPO = "https://github.com/octo/repo"
BASE_ARGS = ["--repo", REPO, "--pr", "5", "--tests", "tests/test_a.py::test_x"]


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Run in an empty dir (no stray config.toml) and restore the ci_fix logger afterwards."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
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


def _result(*statuses: OutcomeStatus) -> FixResult:
    tests = [
        TestOutcome(requested_name=f"t{i}", node_id=f"tests/test_a.py::t{i}", status=s)
        for i, s in enumerate(statuses)
    ]
    return FixResult(
        repo_url=REPO,
        pr_number=5,
        branch="ci-fix/pr-5",
        diff="",
        summary="## ci-fix SUMMARY MARKER",
        tests=tests,
    )


class FakePipeline:
    def __init__(self, result: FixResult | None = None, exc: Exception | None = None) -> None:
        self.result = result
        self.exc = exc
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> FixResult:
        self.calls.append((args, kwargs))
        if self.exc is not None:
            raise self.exc
        assert self.result is not None
        return self.result

    @property
    def settings(self) -> Settings:
        args, kwargs = self.calls[-1]
        return kwargs["settings"]


def _install(monkeypatch: pytest.MonkeyPatch, fake: FakePipeline) -> FakePipeline:
    monkeypatch.setattr(cli, "fix_failing_tests", fake)
    return fake


# ---- slice 0 behaviour still holds ----------------------------------------------------------


def test_version_flag_prints_version_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == "ci-fix 0.1.0"


def test_no_args_is_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_main_none_argv_uses_sys_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["ci-fix"])
    with pytest.raises(SystemExit) as exc:
        main(None)
    assert exc.value.code == 2


def test_unknown_flag_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as exc:
        main([*BASE_ARGS, "--definitely-not-a-flag"])
    assert exc.value.code != 0


@pytest.mark.parametrize(
    "argv",
    [
        ["--pr", "5", "--tests", "t"],
        ["--repo", REPO, "--tests", "t"],
        ["--repo", REPO, "--pr", "5"],
        ["--repo", REPO, "--pr", "5", "--tests"],
        ["--repo", REPO, "--pr", "abc", "--tests", "t"],
    ],
)
def test_missing_or_bad_args_exit_2(argv: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install(monkeypatch, FakePipeline(_result(OutcomeStatus.FIXED)))
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2
    assert fake.calls == []


# ---- running the pipeline -------------------------------------------------------------------


def test_passes_arguments_to_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install(monkeypatch, FakePipeline(_result(OutcomeStatus.FIXED)))
    rc = main(["--repo", REPO, "--pr", "5", "--tests", "a.py::t1", "test_two"])
    assert rc == 0
    assert len(fake.calls) == 1
    args, kwargs = fake.calls[0]
    bound = dict(zip(("repo_url", "pr_number", "failing_tests"), args, strict=False))
    bound.update({k: v for k, v in kwargs.items() if k != "settings"})
    assert bound["repo_url"] == REPO
    assert bound["pr_number"] == 5
    assert list(bound["failing_tests"]) == ["a.py::t1", "test_two"]
    assert isinstance(fake.settings, Settings)
    assert fake.settings.keep_workspace is False


@pytest.mark.parametrize(
    ("statuses", "code"),
    [
        ((OutcomeStatus.FIXED,), 0),
        ((OutcomeStatus.FIXED, OutcomeStatus.ALREADY_PASSING), 0),
        ((OutcomeStatus.ALREADY_PASSING,), 0),
        ((OutcomeStatus.FIXED, OutcomeStatus.UNFIXABLE), 1),
        ((OutcomeStatus.NOT_FOUND,), 1),
        ((OutcomeStatus.AMBIGUOUS, OutcomeStatus.FIXED), 1),
    ],
)
def test_exit_code_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    statuses: tuple[OutcomeStatus, ...],
    code: int,
) -> None:
    _install(monkeypatch, FakePipeline(_result(*statuses)))
    assert main(BASE_ARGS) == code
    assert "SUMMARY MARKER" in capsys.readouterr().out


@pytest.mark.parametrize(
    "exc",
    [
        GitError("clone failed"),
        GitHubError("PR not found"),
        TestEnvError("uv venv failed"),
        TestRunError("pytest crashed"),
        ConfigError("bad config"),
    ],
)
def test_known_errors_exit_2_with_one_line_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], exc: Exception
) -> None:
    _install(monkeypatch, FakePipeline(exc=exc))
    assert main(BASE_ARGS) == 2
    err = capsys.readouterr().err
    assert str(exc) in err
    assert "Traceback" not in err
    message_lines = [line for line in err.splitlines() if str(exc) in line]
    assert len(message_lines) == 1


def test_missing_config_file_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    fake = _install(monkeypatch, FakePipeline(_result(OutcomeStatus.FIXED)))
    assert main([*BASE_ARGS, "--config", str(tmp_path / "nope.toml")]) == 2
    assert "nope.toml" in capsys.readouterr().err
    assert fake.calls == []


def test_config_file_is_loaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "custom.toml"
    cfg.write_text("[ci_fix]\nmax_attempts = 5\n", encoding="utf-8")
    fake = _install(monkeypatch, FakePipeline(_result(OutcomeStatus.FIXED)))
    assert main([*BASE_ARGS, "--config", str(cfg)]) == 0
    assert fake.settings.max_attempts == 5


def test_keep_workspace_flag_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install(monkeypatch, FakePipeline(_result(OutcomeStatus.FIXED)))
    assert main([*BASE_ARGS, "--keep-workspace"]) == 0
    assert fake.settings.keep_workspace is True


def test_log_options_pass_through(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = _install(monkeypatch, FakePipeline(_result(OutcomeStatus.FIXED)))
    log_file = tmp_path / "logs" / "run.log"
    assert main([*BASE_ARGS, "--log-level", "DEBUG", "--log-file", str(log_file)]) == 0
    assert fake.settings.log_level == "DEBUG"
    assert fake.settings.log_file == log_file


def test_invalid_log_level_is_usage_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, FakePipeline(_result(OutcomeStatus.FIXED)))
    with pytest.raises(SystemExit) as exc:
        main([*BASE_ARGS, "--log-level", "LOUD"])
    assert exc.value.code == 2


# ---- unexpected errors and interrupts -------------------------------------------------------


def test_unexpected_error_exits_2_with_one_line_message(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(monkeypatch, FakePipeline(exc=RuntimeError("kaboom\nsecond line")))
    assert main(BASE_ARGS) == 2
    err = capsys.readouterr().err
    assert err.strip() == "ci-fix: unexpected error: RuntimeError: kaboom"
    assert "Traceback" not in err


def test_unexpected_error_traceback_logged_at_debug(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch, FakePipeline(exc=RuntimeError("kaboom")))
    log_file = tmp_path / "run.log"
    assert main([*BASE_ARGS, "--log-file", str(log_file)]) == 2
    text = log_file.read_text(encoding="utf-8")
    assert "Traceback" in text and "kaboom" in text


def test_keyboard_interrupt_exits_130(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(monkeypatch, FakePipeline(exc=KeyboardInterrupt()))  # type: ignore[arg-type]
    assert main(BASE_ARGS) == 130
    assert "interrupted" in capsys.readouterr().err
