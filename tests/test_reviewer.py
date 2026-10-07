"""Tests for ci_fix.guards.reviewer.ClaudeReviewer (slice 5) with fake chat models (no network)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage

from ci_fix.config import Settings
from ci_fix.guards.reviewer import ClaudeReviewer, ReviewVerdict, TestChangeReviewer
from ci_fix.models import FixerFatalError, FixRequest

DIFF = (
    "--- a/tests/test_ops.py\n+++ b/tests/test_ops.py\n"
    "-    assert subtract(5, 3) == 8\n+    assert subtract(5, 3) == 2\n"
)
EXPLANATION = "Root cause: the expected value was wrong.\nTest change: 5 - 3 is 2, not 8."


def _request(tmp_path: Path) -> FixRequest:
    return FixRequest(
        node_id="tests/test_ops.py::test_subtract",
        repo_path=tmp_path,
        attempt=1,
        max_attempts=3,
        failure_message="assert 2 == 8",
        failure_details="AssertionError: assert 2 == 8",
    )


def _text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    parts = []
    for m in messages:
        content = getattr(m, "content", m)
        if isinstance(content, (list, tuple)):
            content = " ".join(
                str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content
            )
        parts.append(str(content))
    return "\n".join(parts)


class _Structured:
    def __init__(self, owner: _StructuredModel) -> None:
        self.owner = owner

    def invoke(self, messages: Any) -> ReviewVerdict:
        self.owner.sent.append(messages)
        if self.owner.exc is not None:
            raise self.owner.exc
        return self.owner.verdict


class _StructuredModel:
    """Fake chat model that supports ``with_structured_output``."""

    def __init__(self, verdict: ReviewVerdict | None = None, exc: Exception | None = None):
        self.verdict = verdict
        self.exc = exc
        self.sent: list[Any] = []
        self.schemas: list[Any] = []

    def with_structured_output(self, schema: Any, **kwargs: Any) -> _Structured:
        self.schemas.append(schema)
        return _Structured(self)

    def bind_tools(self, tools: Any, **kwargs: Any) -> _StructuredModel:  # pragma: no cover
        return self

    def invoke(self, messages: Any) -> Any:  # pragma: no cover - structured path expected
        raise AssertionError("expected the structured-output path")


class _PlainModel:
    """Fake chat model without ``with_structured_output``: replies with a JSON AIMessage."""

    def __init__(self, content: str = "", exc: Exception | None = None) -> None:
        self.content = content
        self.exc = exc
        self.sent: list[Any] = []

    def invoke(self, messages: Any) -> AIMessage:
        self.sent.append(messages)
        if self.exc is not None:
            raise self.exc
        return AIMessage(content=self.content)


class _ApiError(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": message},
        }


SETTINGS = Settings(anthropic_api_key="sk-test-not-real")


def test_review_verdict_model() -> None:
    verdict = ReviewVerdict(approved=False, reason="weakens the test")
    assert (verdict.approved, verdict.reason) == (False, "weakens the test")


def test_settings_review_test_changes_defaults_to_false() -> None:
    assert Settings().review_test_changes is False
    assert Settings(review_test_changes=True).review_test_changes is True


def test_claude_reviewer_satisfies_protocol(tmp_path: Path) -> None:
    reviewer: TestChangeReviewer = ClaudeReviewer(SETTINGS, llm=_StructuredModel())
    assert callable(reviewer.review)


@pytest.mark.parametrize("approved", [True, False])
def test_structured_output_verdict_is_returned(tmp_path: Path, approved: bool) -> None:
    llm = _StructuredModel(ReviewVerdict(approved=approved, reason="because"))
    verdict = ClaudeReviewer(SETTINGS, llm=llm).review(_request(tmp_path), DIFF, EXPLANATION)
    assert isinstance(verdict, ReviewVerdict)
    assert (verdict.approved, verdict.reason) == (approved, "because")
    assert llm.schemas and llm.schemas[0] is ReviewVerdict
    (sent,) = llm.sent
    prompt = _text(sent)
    assert "assert subtract(5, 3) == 2" in prompt  # the diff is shown to the reviewer
    assert "Test change: 5 - 3 is 2" in prompt  # and so is the explanation
    assert "test_subtract" in prompt


@pytest.mark.parametrize("approved", [True, False])
def test_plain_json_reply_is_parsed(tmp_path: Path, approved: bool) -> None:
    content = '{"approved": %s, "reason": "looked at it"}' % ("true" if approved else "false")
    llm = _PlainModel(content)
    verdict = ClaudeReviewer(SETTINGS, llm=llm).review(_request(tmp_path), DIFF, EXPLANATION)
    assert (verdict.approved, verdict.reason) == (approved, "looked at it")
    assert len(llm.sent) == 1
    assert "assert subtract(5, 3) == 2" in _text(llm.sent[0])


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_fatal_api_error_structured_raises_fixer_fatal_error(tmp_path: Path, status: int) -> None:
    llm = _StructuredModel(exc=_ApiError(status, "Your credit balance is too low"))
    with pytest.raises(FixerFatalError, match="credit balance is too low"):
        ClaudeReviewer(SETTINGS, llm=llm).review(_request(tmp_path), DIFF, EXPLANATION)


def test_fatal_api_error_plain_raises_fixer_fatal_error(tmp_path: Path) -> None:
    llm = _PlainModel(exc=_ApiError(400, "temperature is not supported"))
    with pytest.raises(FixerFatalError, match="temperature is not supported"):
        ClaudeReviewer(SETTINGS, llm=llm).review(_request(tmp_path), DIFF, EXPLANATION)


def test_transient_api_error_is_not_fatal(tmp_path: Path) -> None:
    llm = _StructuredModel(exc=_ApiError(529, "overloaded"))
    with pytest.raises(Exception) as info:
        ClaudeReviewer(SETTINGS, llm=llm).review(_request(tmp_path), DIFF, EXPLANATION)
    assert not isinstance(info.value, FixerFatalError)


class _FakeChatAnthropic:
    captured: dict[str, Any] = {}

    def __init__(self, **kwargs: Any) -> None:
        _FakeChatAnthropic.captured = dict(kwargs)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        model = _StructuredModel(ReviewVerdict(approved=True, reason="ok"))
        return model.with_structured_output(schema)

    def bind_tools(self, tools: Any, **kwargs: Any) -> _FakeChatAnthropic:
        return self

    def invoke(self, messages: Any) -> AIMessage:
        return AIMessage(content='{"approved": true, "reason": "ok"}')


def _build_with_fake_chat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, settings: Settings
) -> dict[str, Any]:
    import ci_fix.agent.llm as llm_module

    _FakeChatAnthropic.captured = {}
    monkeypatch.setattr(llm_module, "ChatAnthropic", _FakeChatAnthropic)
    reviewer = ClaudeReviewer(settings)
    verdict = reviewer.review(_request(tmp_path), DIFF, EXPLANATION)  # model may be built lazily
    assert verdict.approved is True
    return _FakeChatAnthropic.captured


def test_temperature_none_is_not_sent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = Settings(anthropic_api_key="sk-x", model="claude-sonnet-5-5", temperature=None)
    captured = _build_with_fake_chat(monkeypatch, tmp_path, settings)
    assert captured["model"] == "claude-sonnet-5-5"
    assert "temperature" not in captured


def test_default_temperature_zero_is_sent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = _build_with_fake_chat(monkeypatch, tmp_path, Settings(anthropic_api_key="sk-x"))
    assert captured["temperature"] == 0.0
    assert captured["model"] == "claude-sonnet-4-6"


# ---- diff ordering / truncation ----------------------------------------------------------


def _chunk(path: str, body: str) -> str:
    return f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n{body}\n"


def test_test_hunks_come_first_and_are_never_truncated(tmp_path: Path) -> None:
    from ci_fix.guards.reviewer import build_review_message

    source = _chunk("src/calc/ops.py", "+" + "s" * 5000)
    test = _chunk("tests/test_ops.py", "+" + "t" * 3000)
    message = build_review_message(_request(tmp_path), source + test, EXPLANATION, 1000)
    assert "t" * 3000 in message  # over the budget on its own, still complete
    assert message.index("tests/test_ops.py") < message.index("src/calc/ops.py")
    assert "s" * 5000 not in message
    assert "source changes truncated" in message


def test_source_hunks_fit_in_the_remaining_budget(tmp_path: Path) -> None:
    from ci_fix.guards.reviewer import order_diff

    source = _chunk("src/calc/ops.py", "+small")
    test = _chunk("tests/test_ops.py", "+x")
    out = order_diff(source + test, 10_000)
    assert out == test + source


def test_split_diff_per_file() -> None:
    from ci_fix.guards.reviewer import split_diff

    diff = _chunk("a.py", "+1") + _chunk("tests/data/x.json", "+2")
    assert [p for p, _ in split_diff(diff)] == ["a.py", "tests/data/x.json"]
    assert "".join(c for _, c in split_diff(diff)) == diff
