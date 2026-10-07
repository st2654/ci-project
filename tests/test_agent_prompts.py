"""Tests for ci_fix.agent.prompts (slice 4): the system prompt and the per-request message."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ci_fix.agent.prompts import SYSTEM_PROMPT, build_user_message
from ci_fix.models import FixAttempt, FixRequest

TARGET = "tests/test_ops.py::test_subtract"
OTHER = "tests/test_ops.py::test_mean"


def make_request(**overrides) -> FixRequest:
    values = dict(
        node_id=TARGET,
        repo_path=Path("/tmp/repo"),
        attempt=2,
        max_attempts=3,
        failure_message="AssertionError: assert 8 == 2",
        failure_details=(
            "def test_subtract():\n>       assert subtract(5, 3) == 2\nE  assert 8 == 2"
        ),
        previous_attempts=[],
        pr_title="Add subtract helper",
        pr_body="This PR adds a subtract function to the calculator.",
        pr_diff="diff --git a/src/calc/ops.py b/src/calc/ops.py\n+def subtract(a, b):\n",
        other_failing_tests=[OTHER],
    )
    values.update(overrides)
    return FixRequest(**values)


# --------------------------------------------------------------------------- #
# SYSTEM_PROMPT
# --------------------------------------------------------------------------- #


def test_system_prompt_is_nonempty_string() -> None:
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 200


@pytest.mark.parametrize("phrase", ["skip", "xfail", "assert", "unfixable"])
def test_system_prompt_integrity_phrases(phrase: str) -> None:
    assert phrase in SYSTEM_PROMPT.lower()


def test_system_prompt_forbids_special_casing_tests() -> None:
    text = SYSTEM_PROMPT.lower()
    assert re.search(r"special[- ]?cas|hard[- ]?cod", text)


def test_system_prompt_forbids_deleting_or_weakening_tests() -> None:
    text = SYSTEM_PROMPT.lower()
    assert re.search(r"delet|remov", text)
    assert re.search(r"weaken|loosen", text)


def test_system_prompt_is_conservative_about_source_changes() -> None:
    text = SYSTEM_PROMPT.lower()
    assert "source" in text
    assert re.search(r"conservative|smallest|minimal", text)


def test_system_prompt_mentions_intent_ambiguous() -> None:
    assert "intent ambiguous" in SYSTEM_PROMPT.lower()


def test_system_prompt_explanation_format() -> None:
    assert "Root cause:" in SYSTEM_PROMPT
    assert "Fix:" in SYSTEM_PROMPT
    assert "Why this file:" in SYSTEM_PROMPT


def test_system_prompt_mentions_finish_tool() -> None:
    assert "finish" in SYSTEM_PROMPT


def test_system_prompt_has_no_secret_placeholders() -> None:
    assert "ANTHROPIC_API_KEY" not in SYSTEM_PROMPT
    assert "sk-ant" not in SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# build_user_message
# --------------------------------------------------------------------------- #


def test_user_message_contains_all_sections() -> None:
    msg = build_user_message(make_request())
    assert isinstance(msg, str)
    assert TARGET in msg
    assert re.search(r"2\s*(/|of)\s*3", msg)
    assert "AssertionError: assert 8 == 2" in msg
    assert "assert subtract(5, 3) == 2" in msg
    assert "Add subtract helper" in msg
    assert "adds a subtract function" in msg
    assert "+def subtract(a, b):" in msg
    assert OTHER in msg


def test_user_message_without_optional_context() -> None:
    msg = build_user_message(
        make_request(pr_title="", pr_body="", pr_diff="", other_failing_tests=[], attempt=1)
    )
    assert TARGET in msg
    assert re.search(r"1\s*(/|of)\s*3", msg)
    assert OTHER not in msg


def test_user_message_defaults_for_new_fields() -> None:
    request = FixRequest(
        node_id=TARGET,
        repo_path=Path("/tmp/repo"),
        attempt=1,
        max_attempts=3,
        failure_message="boom",
        failure_details="",
    )
    assert request.pr_title == ""
    assert request.pr_body == ""
    assert request.pr_diff == ""
    assert request.other_failing_tests == []
    assert request.run_test is None
    msg = build_user_message(request)
    assert TARGET in msg
    assert "boom" in msg


def test_user_message_truncates_long_details() -> None:
    details = "DETAILS_HEAD\n" + "x" * 50_000 + "\nDETAILS_TAIL"
    msg = build_user_message(make_request(failure_details=details))
    assert details not in msg
    assert len(msg) < 30_000
    assert "DETAILS_HEAD" in msg or "DETAILS_TAIL" in msg  # keeps part of it


def test_user_message_keeps_short_details_whole() -> None:
    details = "line\n" * 1000  # 5000 chars, under the ~8000 limit
    msg = build_user_message(make_request(failure_details=details))
    assert details in msg


def test_user_message_truncates_long_body() -> None:
    body = "BODY_HEAD " + "b" * 100_000
    msg = build_user_message(make_request(pr_body=body))
    assert body not in msg
    assert "BODY_HEAD" in msg
    assert len(msg) < 60_000


def test_user_message_truncates_huge_diff() -> None:
    diff = "diff --git a/big b/big\n" + "+y\n" * 100_000  # 300k chars
    msg = build_user_message(make_request(pr_diff=diff))
    assert diff not in msg
    assert "diff --git a/big b/big" in msg
    assert len(msg) < 150_000


def test_user_message_lists_previous_attempts_with_rejections() -> None:
    previous = [
        FixAttempt(
            node_id=TARGET,
            attempt=1,
            outcome="changed",
            explanation="Changed subtract to multiply",
            files_changed=["src/calc/ops.py"],
            accepted=False,
            rejection_reason="target still failing: assert 15 == 2",
        ),
        FixAttempt(
            node_id=TARGET,
            attempt=2,
            outcome="no_change",
            explanation="Gave up reading",
            accepted=False,
            rejection_reason="the fixer made no changes",
        ),
    ]
    msg = build_user_message(make_request(attempt=3, previous_attempts=previous))
    assert "Changed subtract to multiply" in msg
    assert "target still failing: assert 15 == 2" in msg
    assert "the fixer made no changes" in msg
    assert "src/calc/ops.py" in msg
    assert msg.index("Changed subtract to multiply") < msg.index("Gave up reading")


def test_user_message_lists_every_other_failing_test() -> None:
    others = [OTHER, "tests/test_ops.py::TestDivide::test_divide_by_zero"]
    msg = build_user_message(make_request(other_failing_tests=others))
    for other in others:
        assert other in msg
