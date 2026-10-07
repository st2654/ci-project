"""Tests for ci_fix.logging_setup and progress/troubleshooting logs (slice 1)."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from conftest import REPO_URL, FakeRemote, fake_client, pr_info

import ci_fix
from ci_fix.config import Settings
from ci_fix.logging_setup import (
    LOGGER_NAME,
    RedactSecretsFilter,
    configure_logging,
    get_logger,
)
from ci_fix.tools.git import GitError, run_git
from ci_fix.workspace import prepare_pr_checkout


@pytest.fixture(autouse=True)
def _reset_ci_fix_logger() -> Iterator[None]:
    logger = logging.getLogger(LOGGER_NAME)
    saved = (list(logger.handlers), logger.level, logger.propagate)
    yield
    for h in list(logger.handlers):
        if getattr(h, "_ci_fix", False):
            logger.removeHandler(h)
            h.close()
    logger.handlers[:] = saved[0]
    logger.setLevel(saved[1])
    logger.propagate = saved[2]


def test_package_installs_null_handler_and_exports_configure() -> None:
    handlers = logging.getLogger(LOGGER_NAME).handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)
    assert ci_fix.configure_logging is configure_logging


def test_get_logger_namespaces_under_ci_fix() -> None:
    assert get_logger("ci_fix.tools.git").name == "ci_fix.tools.git"
    assert get_logger("other").name == "ci_fix.other"


def test_configure_logging_console_level(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    log = get_logger("ci_fix.test")
    log.info("visible progress")
    log.debug("hidden detail")
    err = capsys.readouterr().err
    assert "visible progress" in err
    assert "hidden detail" not in err


def test_configure_logging_is_idempotent(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    configure_logging("INFO")
    get_logger("ci_fix.test").info("once")
    assert capsys.readouterr().err.count("once") == 1


def test_log_file_gets_debug_while_console_stays_info(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log_file = tmp_path / "logs" / "ci-fix.log"
    configure_logging("INFO", log_file=log_file)
    log = get_logger("ci_fix.test")
    log.debug("debug detail")
    log.info("info progress")
    for h in logging.getLogger(LOGGER_NAME).handlers:
        h.flush()
    content = log_file.read_text()
    assert "debug detail" in content and "info progress" in content
    assert "ci_fix.test" in content
    assert "debug detail" not in capsys.readouterr().err


def test_secrets_are_redacted(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log_file = tmp_path / "ci-fix.log"
    configure_logging("DEBUG", log_file=log_file, secrets=["tok-123", ""])
    get_logger("ci_fix.test").info("using %s now", "tok-123")
    for h in logging.getLogger(LOGGER_NAME).handlers:
        h.flush()
    assert "tok-123" not in capsys.readouterr().err
    assert "tok-123" not in log_file.read_text()
    assert "***" in log_file.read_text()


def test_redact_filter_without_secrets_leaves_record_alone() -> None:
    record = logging.LogRecord("ci_fix", logging.INFO, __file__, 1, "x=%s", ("y",), None)
    assert RedactSecretsFilter([]).filter(record)
    assert record.args == ("y",)


def test_run_git_logs_commands_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    run_git(["--version"])
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("$ git --version" in m for m in messages)
    assert any("git ok in" in m for m in messages)


def test_run_git_failure_log_is_redacted(caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    with pytest.raises(GitError):
        run_git(["rev-parse", "s3cr3t-tok"], cwd=tmp_path, token="s3cr3t-tok")
    assert "s3cr3t-tok" not in caplog.text
    assert "git exited" in caplog.text


def test_settings_log_defaults_and_validation() -> None:
    s = Settings()
    assert s.log_level == "INFO"
    assert s.log_file is None
    assert "~" not in str(Settings(log_file=Path("~/x.log")).log_file)
    with pytest.raises(ValueError):
        Settings(log_level="LOUD")


def test_prepare_logs_progress_steps(
    fake_remote: FakeRemote, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    settings = Settings(workspace_dir=tmp_path / "ws")
    client = fake_client(pr_info(fake_remote))
    prepare_pr_checkout(REPO_URL, fake_remote.pr_number, settings, client, str(fake_remote.bare))
    text = caplog.text
    for step in ("[setup 1/4]", "[setup 2/4]", "[setup 3/4]", "[setup 4/4]", "Workspace ready"):
        assert step in text
    assert "Cloned in" in text
    assert f"Creating patch branch ci-fix/pr-{fake_remote.pr_number}" in text
    # Tool-level detail stays at DEBUG so INFO is not noisy.
    assert "Clone finished" not in text


def test_exception_tracebacks_are_redacted(tmp_path: Path) -> None:
    log_file = tmp_path / "ci-fix.log"
    configure_logging("INFO", log_file=log_file, secrets=["tok-xyz"])
    try:
        raise RuntimeError("auth failed for tok-xyz")
    except RuntimeError:
        get_logger("ci_fix.test").exception("boom")
    for h in logging.getLogger(LOGGER_NAME).handlers:
        h.flush()
    content = log_file.read_text()
    assert "RuntimeError" in content
    assert "tok-xyz" not in content


def test_clone_url_credentials_not_logged(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from ci_fix.tools.git import GitRepo

    caplog.set_level(logging.DEBUG, logger=LOGGER_NAME)
    with pytest.raises(GitError) as exc:
        GitRepo.clone("https://user:hunter2@127.0.0.1:9/none.git", tmp_path / "c")
    assert "hunter2" not in caplog.text
    assert "hunter2" not in str(exc.value)
