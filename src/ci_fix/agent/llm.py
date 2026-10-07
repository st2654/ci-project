"""Shared construction and error handling for the Claude chat model (fixer and reviewer)."""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel

from ci_fix.config import ConfigError, Settings
from ci_fix.models import FixerFatalError

# HTTP statuses that mean "this request will never succeed": invalid request (incl.
# unsupported parameters and low credit balance), auth, permission, unknown model.
_FATAL_STATUS = {400, 401, 403, 404}


def build_chat_model(settings: Settings, purpose: str = "the Claude fixer") -> BaseChatModel:
    """A ``ChatAnthropic`` built from ``settings``; ``temperature`` is sent only if set.

    Raises ConfigError when ``ANTHROPIC_API_KEY`` is missing (``purpose`` names the caller).
    """
    if settings.anthropic_api_key is None:
        raise ConfigError(
            f"Missing required environment variable: ANTHROPIC_API_KEY (needed by {purpose})"
        )
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": settings.max_output_tokens,
        "api_key": settings.anthropic_api_key.get_secret_value(),
        "max_retries": 3,
        "timeout": 120,
    }
    if settings.temperature is not None:  # some models reject `temperature`
        kwargs["temperature"] = settings.temperature
    return ChatAnthropic(**kwargs)


def is_fatal_api_error(exc: Exception) -> bool:
    """True if ``exc`` is an API error that no retry can fix (400/401/403/404)."""
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status in _FATAL_STATUS


def api_error_message(exc: Exception) -> str:
    """The API's own error message if present, else the first line of ``str(exc)``."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    return str(exc).splitlines()[0][:300]


def invoke_or_fatal(llm: Any, messages: Any) -> Any:
    """``llm.invoke(messages)``; fatal API errors become :class:`FixerFatalError`."""
    try:
        return llm.invoke(messages)
    except Exception as exc:
        if is_fatal_api_error(exc):
            raise FixerFatalError(
                f"Claude API rejected the request ({type(exc).__name__}): {api_error_message(exc)}"
            ) from exc
        raise
