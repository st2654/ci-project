"""ci-fix: fix failing tests on a GitHub pull request with an LLM agent."""

import logging

from ci_fix.agent import ClaudeFixer
from ci_fix.config import ConfigError, Settings, load_settings
from ci_fix.logging_setup import configure_logging
from ci_fix.models import (
    FixAttempt,
    Fixer,
    FixerFatalError,
    FixRequest,
    FixResult,
    NoOpFixer,
    OutcomeStatus,
    TestOutcome,
)
from ci_fix.pipeline import fix_failing_tests

# Library default: silent unless the application configures logging.
logging.getLogger("ci_fix").addHandler(logging.NullHandler())

__version__ = "0.1.0"

__all__ = [
    "ClaudeFixer",
    "ConfigError",
    "FixAttempt",
    "FixRequest",
    "FixResult",
    "Fixer",
    "FixerFatalError",
    "NoOpFixer",
    "OutcomeStatus",
    "Settings",
    "TestOutcome",
    "__version__",
    "configure_logging",
    "fix_failing_tests",
    "load_settings",
]
