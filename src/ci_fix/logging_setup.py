"""Logging for ci-fix.

The library logs under the ``ci_fix`` logger and never configures handlers on import.
Applications (the CLI, or a caller) call :func:`configure_logging` once at startup.

Levels:
- INFO: pipeline progress a user wants to watch (cloning, fetching PR, branch created, ...).
- DEBUG: troubleshooting detail (every git command, its duration and stderr).
- WARNING/ERROR: something went wrong or needs attention.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable
from pathlib import Path

LOGGER_NAME = "ci_fix"
CONSOLE_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"
_REDACTED = "***"


class RedactSecretsFilter(logging.Filter):
    """Replace known secret values in log messages with ``***``."""

    def __init__(self, secrets: Iterable[str]) -> None:
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def _redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, _REDACTED)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._secrets:
            return True
        record.msg, record.args = self._redact(record.getMessage()), None
        if record.exc_info:
            # Pre-render the traceback so exception text is masked too.
            record.exc_text = self._redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = self._redact(record.exc_text)
        if record.stack_info:
            record.stack_info = self._redact(record.stack_info)
        return True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``ci_fix`` namespace (pass ``__name__``)."""
    return logging.getLogger(name if name.startswith(LOGGER_NAME) else f"{LOGGER_NAME}.{name}")


def configure_logging(
    level: str | int = "INFO",
    log_file: Path | None = None,
    secrets: Iterable[str] = (),
) -> logging.Logger:
    """Send ``ci_fix`` logs to stderr at ``level`` and, optionally, everything (DEBUG) to a file.

    Calling it again replaces the handlers it added before. ``secrets`` are masked in output.
    """
    logger = logging.getLogger(LOGGER_NAME)
    for handler in [h for h in logger.handlers if getattr(h, "_ci_fix", False)]:
        logger.removeHandler(handler)
        handler.close()

    redact = RedactSecretsFilter(secrets)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT, DATE_FORMAT))
    console.addFilter(redact)
    console._ci_fix = True  # type: ignore[attr-defined]
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file).expanduser()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
        file_handler.addFilter(redact)
        file_handler._ci_fix = True  # type: ignore[attr-defined]
        logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG if log_file is not None else level)
    logger.propagate = False
    return logger
