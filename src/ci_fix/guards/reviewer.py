"""Optional second opinion on test-file changes: a strict LLM reviewer.

Enabled with ``review_test_changes = true``. It runs only for attempts that passed the
deterministic patch checker and verification and that changed a test file.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from ci_fix.agent.llm import build_chat_model, invoke_or_fatal
from ci_fix.config import Settings
from ci_fix.guards.test_layout import DEFAULT_LAYOUT
from ci_fix.logging_setup import get_logger
from ci_fix.models import FixRequest

log = get_logger(__name__)

FAILURE_MAX_CHARS = 6000
EXPLANATION_MAX_CHARS = 4000

REVIEWER_SYSTEM_PROMPT = """\
You are a strict code reviewer guarding test integrity. An automated fixer changed files to
make a failing pytest test pass, and its change included edits to TEST files.

Decide whether EVERY test-file change in the diff is legitimate:
- Legitimate: the old expectation was demonstrably wrong (contradicts the documented or
  clearly intended behaviour, a typo, a broken fixture), and the change still verifies what
  the test is meant to verify.
- Not legitimate: anything that weakens or circumvents the test — loosening or removing
  assertions, changing expected values to match buggy output, expecting the error instead of
  fixing it, skipping, catching exceptions, special-casing inputs, or bending fixtures so the
  code under test is no longer really exercised.

Reject when in doubt. Reply with `approved` (true/false) and a one-sentence `reason`.
If your reply is plain text, reply with only a JSON object:
{"approved": <true|false>, "reason": "<one sentence>"}
"""


class ReviewVerdict(BaseModel):
    """The reviewer's decision on the test-file changes of one attempt."""

    approved: bool
    reason: str


class TestChangeReviewer(Protocol):
    """Approves or rejects the test-file changes of an attempt that otherwise passed."""

    __test__ = False

    def review(self, request: FixRequest, diff: str, explanation: str) -> ReviewVerdict: ...


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n[... truncated {len(text) - max_chars} chars]"


_FILE_HEADER_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.MULTILINE)


def split_diff(diff: str) -> list[tuple[str, str]]:
    """``[(path, chunk)]``, one chunk per file of a ``git diff``."""
    starts = [(m.start(), m.group(2)) for m in _FILE_HEADER_RE.finditer(diff)]
    if not starts:
        return [("", diff)] if diff else []
    chunks = []
    if starts[0][0] > 0:
        chunks.append(("", diff[: starts[0][0]]))
    for i, (start, path) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(diff)
        chunks.append((path, diff[start:end]))
    return chunks


def order_diff(diff: str, max_chars: int) -> str:
    """Test-file chunks first and never truncated; source chunks fill what is left of the
    ``max_chars`` budget and are truncated if needed."""
    chunks = split_diff(diff)
    tests = "".join(c for p, c in chunks if p and DEFAULT_LAYOUT.is_test_path(p))
    source = "".join(c for p, c in chunks if not (p and DEFAULT_LAYOUT.is_test_path(p)))
    budget = max(0, max_chars - len(tests))
    if len(source) > budget:
        files = ", ".join(p for p, _ in chunks if p and not DEFAULT_LAYOUT.is_test_path(p))
        source = (
            f"{source[:budget]}\n[... source changes truncated: {len(source) - budget} chars;"
            f" source files changed: {files or 'unknown'}]\n"
        )
    return tests + source


def build_review_message(
    request: FixRequest, diff: str, explanation: str, diff_max_chars: int
) -> str:
    """The reviewer's task: failing test, its failure, the fixer's explanation and the diff.

    Test-file hunks come first and are never truncated (they are what is being judged).
    """
    failure = f"{request.failure_message}\n{request.failure_details}".strip()
    return "\n".join(
        [
            f"# Failing test\n`{request.node_id}`",
            "",
            "# Failure before the fix",
            "```",
            _truncate(failure, FAILURE_MAX_CHARS) or "(no details)",
            "```",
            "",
            "# Fixer's explanation",
            _truncate(explanation, EXPLANATION_MAX_CHARS) or "(none)",
            "",
            "# Diff of the fix",
            "```diff",
            order_diff(diff, diff_max_chars).rstrip("\n"),
            "```",
        ]
    )


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict(raw: Any) -> ReviewVerdict:
    """A ReviewVerdict from structured output (model or dict) or a text reply with JSON."""
    if isinstance(raw, ReviewVerdict):
        return raw
    if isinstance(raw, dict):
        return ReviewVerdict.model_validate(raw)
    content = getattr(raw, "content", raw)
    if isinstance(content, list):  # content blocks
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    match = _JSON_OBJECT_RE.search(str(content))
    if match is None:
        raise ValueError(f"reviewer reply has no JSON verdict: {str(content)[:200]!r}")
    try:
        return ReviewVerdict.model_validate(json.loads(match.group(0)))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"reviewer reply is not a valid verdict: {exc}") from exc


class ClaudeReviewer:
    """``TestChangeReviewer`` backed by one structured chat-model call (Claude by default)."""

    def __init__(self, settings: Settings, llm: BaseChatModel | None = None) -> None:
        self.settings = settings
        self.llm = llm if llm is not None else build_chat_model(settings, "the test reviewer")
        try:
            self.structured: Any = self.llm.with_structured_output(ReviewVerdict)
        except (NotImplementedError, AttributeError):  # e.g. simple fake chat models
            self.structured = None

    def review(self, request: FixRequest, diff: str, explanation: str) -> ReviewVerdict:
        messages = [
            SystemMessage(REVIEWER_SYSTEM_PROMPT),
            HumanMessage(
                build_review_message(request, diff, explanation, self.settings.pr_diff_max_chars)
            ),
        ]
        raw = invoke_or_fatal(self.structured or self.llm, messages)
        verdict = parse_verdict(raw)
        log.info(
            "[review] %s: %s — %s",
            request.node_id,
            "approved" if verdict.approved else "rejected",
            verdict.reason,
        )
        return verdict
