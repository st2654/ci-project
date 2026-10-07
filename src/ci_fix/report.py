"""Commit message, PR description and PR comments for a ci-fix run (pure functions).

Everything here only formats data; nothing touches git or the network. The texts must be
readable in 2–3 minutes: short sections, one excerpt per fix, empty sections left out.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, Field

from ci_fix.models import OutcomeStatus, TestOutcome

COMMENT_MARKER = "<!-- ci-fix -->"
PR_TITLE_MAX = 120
REASON_MAX = 300  # chars of an unfixable reason shown
COMMENT_REASON_MAX = 120  # chars of a reason in a PR comment line
EXCERPT_LINES = 3  # explanation lines shown per fix
LINE_MAX = 200  # chars per explanation line
LIST_MAX = 20  # pre-existing failures listed before "... and N more"
SOURCE_NOTE = "changed source code; developer context may be missing — please review"
TEST_CHANGE_LABEL = "⚠ review: test expectations changed"

_ROOT_CAUSE_RE = re.compile(r"^\s*(?:[-*]\s*)?\**root cause\**\s*:", re.IGNORECASE)
_FIX_RE = re.compile(r"^\s*(?:[-*]\s*)?\**fix\**\s*:", re.IGNORECASE)
_TEST_CHANGE_RE = re.compile(r"^\s*(?:[-*]\s*)?\**test change\**\s*:", re.IGNORECASE)
_SIDE_EFFECT_PREFIX = "fixed by the fix for "
TEXT_MAX = 60_000  # PR body, comment and commit message length cap
ZWSP = "\u200b"

# Sanitizing text that came from the LLM or from test output, so it cannot ping people,
# close issues or inject markup when posted on GitHub.
_MENTION_RE = re.compile(r"@(?=[A-Za-z0-9_/-])")
_CLOSING_RE = re.compile(
    r"\b((?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s+(?:[\w.-]+/[\w.-]+)?)#(?=\d)",
    re.IGNORECASE,
)
# HTML comments and tags of real HTML elements. Python's ``<module>``/``<lambda>`` in
# tracebacks are not HTML and are kept.
_HTML_TAGS = (
    "a|abbr|b|blockquote|body|br|button|code|dd|del|details|div|dl|dt|em|embed|font|form|"
    "h[1-6]|head|hr|html|i|iframe|img|input|ins|kbd|li|link|meta|object|ol|p|picture|pre|"
    "q|s|samp|script|section|select|small|source|span|strike|strong|style|sub|summary|sup|"
    "svg|table|tbody|td|textarea|tfoot|th|thead|tr|tt|u|ul|var|video"
)
_HTML_RE = re.compile(
    rf"<!--.*?(?:-->|$)|</?(?:{_HTML_TAGS})(?:\s[^<>]*)?/?>", re.IGNORECASE | re.DOTALL
)


class ReportData(BaseModel):
    """What the texts are built from (the run's outcomes plus PR metadata)."""

    pr_number: int
    pr_title: str = ""
    pr_url: str = ""  # the original PR
    head_ref: str = ""  # the original PR's branch
    tests: list[TestOutcome] = Field(default_factory=list)  # in requested order
    preexisting_failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    full_suite_checked: bool = False
    reviewer_enabled: bool = False


# --------------------------------------------------------------------------- helpers


def sanitize(text: str) -> str:
    """Neutralize LLM/test-output text for GitHub: HTML, @mentions, closing keywords.

    Strips HTML comments and tags of HTML elements, puts a zero-width space after ``@``
    (no mention) and before the ``#`` of ``fixes #N`` / ``closes owner/repo#N`` (no
    auto-close). Only for untrusted text, never for ci-fix's own strings (the marker).
    """
    text = _HTML_RE.sub("", text)
    text = _MENTION_RE.sub("@" + ZWSP, text)
    return _CLOSING_RE.sub(lambda m: f"{m.group(1)}{ZWSP}#", text)


def cap_text(text: str, limit: int = TEXT_MAX) -> str:
    """``text`` cut at a line boundary to at most ``limit`` chars, with a truncation note."""
    if len(text) <= limit:
        return text
    budget = limit - len(f"… (truncated; {len(text)} more characters)\n")
    cut = text.rfind("\n", 0, max(budget, 0)) + 1  # 0 when there is no line break
    if cut == 0:
        cut = max(budget, 0)
    return f"{text[:cut]}… (truncated; {len(text) - cut} more characters)\n"


def _one_line(text: str, limit: int = LINE_MAX) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _unique(tests: Iterable[TestOutcome]) -> list[TestOutcome]:
    """One outcome per test (several requested names can resolve to the same node id)."""
    seen: set[str] = set()
    out: list[TestOutcome] = []
    for t in tests:
        key = t.node_id or t.requested_name
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def _label(t: TestOutcome) -> str:
    return t.node_id or t.requested_name


def _with(data: ReportData, *statuses: OutcomeStatus) -> list[TestOutcome]:
    return _unique(t for t in data.tests if t.status in statuses)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _side_effect_of(t: TestOutcome) -> str | None:
    """The test whose fix fixed ``t`` as a side effect, or None."""
    if t.reason.startswith(_SIDE_EFFECT_PREFIX):
        return t.reason[len(_SIDE_EFFECT_PREFIX) :]
    return None


def explanation_excerpt(explanation: str) -> list[str]:
    """``Root cause:`` and ``Fix:`` lines of an explanation (else its first 2 lines), ≤3 lines."""
    lines = [ln.strip() for ln in explanation.splitlines() if ln.strip()]
    picked = [ln for ln in lines if _ROOT_CAUSE_RE.match(ln) or _FIX_RE.match(ln)]
    if not picked:
        picked = [ln for ln in lines if not _TEST_CHANGE_RE.match(ln)][:2]
    return [
        _one_line(sanitize(ln.lstrip("-* ").replace("**", ""))) for ln in picked[:EXCERPT_LINES]
    ]


def test_change_justification(explanation: str) -> str:
    """The ``Test change: ...`` line of an explanation ("" if there is none)."""
    for ln in explanation.splitlines():
        if _TEST_CHANGE_RE.match(ln):
            return _one_line(sanitize(ln.strip().lstrip("-* ").replace("**", "")))
    return ""


def _files_touched(fixed: list[TestOutcome]) -> list[str]:
    return sorted({f for t in fixed for f in t.files_changed})


def _summary_sentence(data: ReportData) -> str:
    fixed = _with(data, OutcomeStatus.FIXED)
    unfixable = _with(data, OutcomeStatus.UNFIXABLE)
    files = _files_touched(fixed)
    text = f"Fixed {_plural(len(fixed), 'failing test')}"
    if unfixable:
        text += f"; {len(unfixable)} could not be fixed"
    text += "."
    if files:
        shown = ", ".join(files[:5]) + (f" and {len(files) - 5} more" if len(files) > 5 else "")
        text += f" Changed {_plural(len(files), 'file')}: {shown}."
    return text


# --------------------------------------------------------------------------- sections
# Each returns (title, lines) — empty lines = section omitted. ``code`` wraps ids in
# backticks (markdown) or not (plain git text).


def _code(text: str, md: bool) -> str:
    if not md:
        return text
    if "`" in text:  # a single-backtick span would end early
        return f"`` {text} ``"
    return f"`{text}`"


def _fixed_lines(data: ReportData, md: bool) -> list[str]:
    lines: list[str] = []
    for t in _with(data, OutcomeStatus.FIXED):
        cause = _side_effect_of(t)
        if cause is not None:
            lines.append(f"- {_code(_label(t), md)} — Fixed by the fix for {_code(cause, md)}")
            continue
        lines.append(f"- {_code(_label(t), md)}")
        lines.extend(f"  {ln}" for ln in explanation_excerpt(t.explanation))
        if t.files_changed:
            lines.append(f"  Files: {', '.join(t.files_changed)}")
    return lines


def _test_change_lines(data: ReportData, md: bool) -> list[str]:
    lines: list[str] = []
    for t in _with(data, OutcomeStatus.FIXED):
        if not t.test_changes:
            continue
        for change in t.test_changes:
            lines.append(f"- {_code(sanitize(change), md)}")
        why = test_change_justification(t.explanation)
        if why:
            lines.append(f"  {why}")
    return lines


def _source_lines(data: ReportData, md: bool) -> list[str]:
    return [
        f"- {_code(_label(t), md)}: {SOURCE_NOTE}"
        for t in _with(data, OutcomeStatus.FIXED)
        if t.source_changed
    ]


def _unfixable_lines(data: ReportData, md: bool) -> list[str]:
    lines = []
    for t in _with(data, OutcomeStatus.UNFIXABLE):
        reason = _one_line(sanitize(t.reason or "no reason given"), REASON_MAX)
        lines.append(f"- {_code(_label(t), md)} ({_plural(t.attempts, 'attempt')}): {reason}")
    return lines


def _not_found_lines(data: ReportData, md: bool) -> list[str]:
    seen: set[str] = set()
    lines = []
    for t in data.tests:
        if t.status not in (OutcomeStatus.NOT_FOUND, OutcomeStatus.AMBIGUOUS):
            continue
        if t.requested_name in seen:
            continue
        seen.add(t.requested_name)
        lines.append(
            f"- {_code(t.requested_name, md)}: {_one_line(sanitize(t.reason or t.status.value))}"
        )
    return lines


def _already_passing_lines(data: ReportData, md: bool) -> list[str]:
    return [f"- {_code(_label(t), md)}" for t in _with(data, OutcomeStatus.ALREADY_PASSING)]


def _preexisting_lines(data: ReportData, md: bool) -> list[str]:
    ids = data.preexisting_failures
    lines = [f"- {_code(i, md)}" for i in ids[:LIST_MAX]]
    if len(ids) > LIST_MAX:
        lines.append(f"- … and {len(ids) - LIST_MAX} more")
    return lines


def _warning_lines(data: ReportData, md: bool) -> list[str]:
    return [f"- {_one_line(sanitize(w), REASON_MAX)}" for w in data.warnings]


def _verification_lines(data: ReportData, md: bool) -> list[str]:
    lines = ["- Re-ran all requested tests after every fix"]
    if data.full_suite_checked:
        lines.append("- Full test suite checked for regressions")
    checks = "patch checker"
    if data.reviewer_enabled:
        checks += " + Claude reviewer"
    lines.append(f"- Integrity checks: {checks}")
    return lines


def _sections(data: ReportData, md: bool) -> list[tuple[str, list[str]]]:
    return [
        ("Fixed", _fixed_lines(data, md)),
        (f"Test changes ({TEST_CHANGE_LABEL})", _test_change_lines(data, md)),
        ("Source changes", _source_lines(data, md)),
        ("Unfixable", _unfixable_lines(data, md)),
        ("Not found / ambiguous", _not_found_lines(data, md)),
        ("Already passing (not touched)", _already_passing_lines(data, md)),
        ("Pre-existing failures (not addressed)", _preexisting_lines(data, md)),
        ("Warnings", _warning_lines(data, md)),
        ("Verification", _verification_lines(data, md)),
    ]


def _footer(data: ReportData) -> str:
    url = f" ({data.pr_url})" if data.pr_url else ""
    return f"Generated by ci-fix for #{data.pr_number}{url}."


# --------------------------------------------------------------------------- public API


def build_commit_title(data: ReportData) -> str:
    n = len(_with(data, OutcomeStatus.FIXED))
    return f"ci-fix: fix {_plural(n, 'failing test')} in #{data.pr_number}"


def build_commit_message(data: ReportData) -> str:
    """Squash commit message: title line, then plain-text sections (git-friendly)."""
    parts = [build_commit_title(data), _summary_sentence(data)]
    for title, lines in _sections(data, md=False):
        if lines:
            parts.append("\n".join([f"{title}:", *lines]))
    parts.append(_footer(data))
    return cap_text("\n\n".join(parts) + "\n")


def build_pr_title(pr_number: int, pr_title: str) -> str:
    """``ci-fix: fixes for #N — <title>``, at most 120 chars."""
    text = f"ci-fix: fixes for #{pr_number} — {pr_title}".strip(" —")
    title = _one_line(text, PR_TITLE_MAX)
    return title


def build_pr_body(data: ReportData) -> str:
    """Markdown description of the fix PR (also ``FixResult.summary``)."""
    head = f" (`{data.head_ref}`)" if data.head_ref else ""
    parts = [
        f"Fixes failing tests in #{data.pr_number}{head}.",
        "### Summary\n" + _summary_sentence(data),
    ]
    for title, lines in _sections(data, md=True):
        if lines:
            parts.append("\n".join([f"### {title}", *lines]))
    parts.append(f"---\n{_footer(data)}")
    return cap_text("\n\n".join(parts) + "\n")


def _status_lines(data: ReportData) -> list[str]:
    lines = []
    for t in _unique(data.tests):
        line = f"- `{_label(t)}`: {t.status.value.replace('_', ' ')}"
        if t.status != OutcomeStatus.FIXED and t.reason:
            line += f" — {_one_line(sanitize(t.reason), COMMENT_REASON_MAX)}"
        lines.append(line)
    return lines


def build_fix_pr_comment(fix_pr_url: str, data: ReportData) -> str:
    """Comment for the original PR: link to the fix PR + one line per test."""
    lines = [
        COMMENT_MARKER,
        f"**ci-fix** proposes fixes for the failing tests in {fix_pr_url}",
        "",
        *_status_lines(data),
    ]
    return cap_text("\n".join(lines) + "\n")


def build_no_fix_comment(data: ReportData) -> str:
    """Comment for the original PR when nothing could be fixed (reasons per test)."""
    parts = [COMMENT_MARKER + "\n**ci-fix** could not fix any of the requested tests."]
    for title, lines in (
        ("Unfixable", _unfixable_lines(data, md=True)),
        ("Not found / ambiguous", _not_found_lines(data, md=True)),
        ("Already passing", _already_passing_lines(data, md=True)),
    ):
        if lines:
            parts.append("\n".join([f"### {title}", *lines]))
    parts.append(_footer(data))
    return cap_text("\n\n".join(parts) + "\n")
