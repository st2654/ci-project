"""ci-fix: fix failing tests on a GitHub pull request with an LLM agent."""

import logging

from ci_fix.config import ConfigError, Settings, load_settings
from ci_fix.logging_setup import configure_logging

# Library default: silent unless the application configures logging.
logging.getLogger("ci_fix").addHandler(logging.NullHandler())

__version__ = "0.1.0"

__all__ = ["ConfigError", "Settings", "__version__", "configure_logging", "load_settings"]
