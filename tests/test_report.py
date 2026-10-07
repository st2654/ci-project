"""Tests for ci_fix.report (slice 8): commit message, PR title/body and PR comments."""

from __future__ import annotations

from typing import Any

import pytest

from ci_fix.models import OutcomeStatus, TestOutcome
from ci_fix.report import (
    COMMENT_MARKER,
    TEXT_MAX,
    ZWSP,
    ReportData,
    _code,
    build_commit_message,
    build_fix_pr_comment,
    build_no_fix_comment,
    build_pr_body,
    build_pr_title,
    cap_text,
    explanation_excerpt,
    sanitize,
)

PR = 42
PR_URL = "https://github.com/octo/sample/pull/42"
FIX_PR_URL = "https://github.com/octo/sample/pull/43"
T_SUB = "tests/test_ops.py::test_subtract"
T_DIV = "tests/test_ops.py::TestDivide::test_divide_by_zero"
T_MEAN = "tests/test_ops.py::test_mean"
OPS_PY = "src/calc/ops.py"

EXPLANATION = (
    "I looked at the code for a while.\n"
    "Root cause: subtract() returned a + b.\n"
    "Fix: return a - b instead.\n"
    "Some trailing chatter that should not appear.\n"
)


def fixed(node_id: str = T_SUB, **kw: Any) -> TestOutcome:
    defaults: dict[str, Any] = {
        "requested_name": node_id,
        "node_id": node_id,
        "status": OutcomeStatus.FIXED,
        "attempts": 1,
        "files_changed": [OPS_PY],
        "explanation": EXPLANATION,
        "source_changed": False,
    }
    defaults.update(kw)
    return TestOutcome(**defaults)


def outcome(name: str, status: OutcomeStatus, **kw: Any) -> TestOutcome:
    return TestOutcome(requested_name=name, node_id=kw.pop("node_id", name), status=status, **kw)


def data(*tests: TestOutcome, **kw: Any) -> ReportData:
    defaults: dict[str, Any] = {
        "pr_number": PR,
        "pr_title": "Add calculator ops",
        "pr_url": PR_URL,
        "head_ref": "feature",
        "tests": list(tests),
    }
    defaults.update(kw)
    return ReportData(**defaults)


def title_of(message: str) -> str:
    return message.splitlines()[0]


# ---- titles ---------------------------------------------------------------------------------


def test_pr_title_format() -> None:
    assert build_pr_title(7, "Add ops") == "ci-fix: fixes for #7 — Add ops"


def test_pr_title_truncated_to_120() -> None:
    title = build_pr_title(7, "x" * 500)
    assert len(title) <= 120
    assert title.startswith("ci-fix: fixes for #7 — ")


def test_pr_title_short_title_untouched() -> None:
    title = build_pr_title(123, "y" * 50)
    assert title == "ci-fix: fixes for #123 — " + "y" * 50


def test_commit_title_singular() -> None:
    assert title_of(build_commit_message(data(fixed()))) == f"ci-fix: fix 1 failing test in #{PR}"


def test_commit_title_plural() -> None:
    msg = build_commit_message(data(fixed(T_SUB), fixed(T_DIV)))
    assert title_of(msg) == f"ci-fix: fix 2 failing tests in #{PR}"


def test_commit_title_counts_only_fixed() -> None:
    msg = build_commit_message(
        data(fixed(T_SUB), outcome(T_MEAN, OutcomeStatus.UNFIXABLE, reason="no idea", attempts=3))
    )
    assert title_of(msg) == f"ci-fix: fix 1 failing test in #{PR}"


def test_commit_title_line_then_blank_line() -> None:
    lines = build_commit_message(data(fixed())).splitlines()
    assert lines[1] == ""


# ---- explanation parsing --------------------------------------------------------------------


def test_excerpt_picks_root_cause_and_fix() -> None:
    lines = explanation_excerpt(EXPLANATION)
    assert lines == ["Root cause: subtract() returned a + b.", "Fix: return a - b instead."]


def test_excerpt_fallback_first_two_lines() -> None:
    lines = explanation_excerpt("first line\n\nsecond line\nthird line\n")
    assert lines == ["first line", "second line"]


def test_excerpt_at_most_three_lines() -> None:
    text = "\n".join(["Root cause: a", "Fix: b", "Root cause: c", "Fix: d", "Fix: e"])
    assert len(explanation_excerpt(text)) <= 3


def test_excerpt_handles_markdown_bullets_and_bold() -> None:
    lines = explanation_excerpt("- **Root cause:** x broke\n* **Fix:** y\n")
    assert any(ln.startswith("Root cause:") and "x broke" in ln for ln in lines)
    assert any(ln.startswith("Fix:") for ln in lines)
    assert all("**" not in ln for ln in lines)


def test_excerpt_empty() -> None:
    assert explanation_excerpt("") == []


def test_body_shows_excerpt_not_chatter() -> None:
    body = build_pr_body(data(fixed()))
    assert "Root cause: subtract() returned a + b." in body
    assert "Fix: return a - b instead." in body
    assert "trailing chatter" not in body
    assert "looked at the code" not in body


def test_body_lists_files_per_fix() -> None:
    body = build_pr_body(data(fixed(files_changed=[OPS_PY, "src/calc/util.py"])))
    assert f"Files: {OPS_PY}, src/calc/util.py" in body


def test_side_effect_fix_rendered() -> None:
    side = fixed(T_DIV, reason=f"fixed by the fix for {T_SUB}", explanation="", files_changed=[])
    body = build_pr_body(data(fixed(T_SUB), side))
    assert "Fixed by the fix for" in body
    line = next(ln for ln in body.splitlines() if T_DIV in ln)
    assert "Fixed by the fix for" in line and T_SUB in line


# ---- PR body structure ----------------------------------------------------------------------


def test_body_starts_with_fixes_line() -> None:
    body = build_pr_body(data(fixed()))
    assert body.startswith(f"Fixes failing tests in #{PR} (`feature`).")


def test_body_uses_h3_headers() -> None:
    body = build_pr_body(data(fixed()))
    headers = [ln for ln in body.splitlines() if ln.startswith("#")]
    assert headers, body
    assert all(ln.startswith("### ") for ln in headers)
    assert "### Summary" in body
    assert "### Fixed" in body
    assert "### Verification" in body


def test_commit_body_has_no_markdown_headers() -> None:
    msg = build_commit_message(
        data(
            fixed(source_changed=True, test_changes=["a::b: 1 → 2"]),
            outcome(T_MEAN, OutcomeStatus.UNFIXABLE, reason="r", attempts=3),
            warnings=["w"],
            preexisting_failures=["tests/x.py::y"],
        )
    )
    assert "###" not in msg
    assert "Fixed" in msg and "Unfixable" in msg


def test_empty_sections_omitted() -> None:
    body = build_pr_body(data(fixed()))
    for absent in (
        "Test changes",
        "Source changes",
        "Unfixable",
        "Not found",
        "Pre-existing",
        "Warnings",
    ):
        assert absent not in body, absent


def test_footer() -> None:
    body = build_pr_body(data(fixed()))
    assert f"Generated by ci-fix for #{PR} ({PR_URL})." in body
    msg = build_commit_message(data(fixed()))
    assert f"Generated by ci-fix for #{PR} ({PR_URL})." in msg


def test_summary_mentions_counts_and_files() -> None:
    body = build_pr_body(
        data(fixed(T_SUB), outcome(T_MEAN, OutcomeStatus.UNFIXABLE, reason="r", attempts=3))
    )
    summary = body.split("### Summary", 1)[1].split("###", 1)[0]
    assert "1" in summary and OPS_PY in summary


# ---- test / source changes ------------------------------------------------------------------


def test_test_change_rendered_with_justification() -> None:
    t = fixed(
        test_changes=[f"{T_SUB}: assert subtract(5, 3) == 8 → assert subtract(5, 3) == 2"],
        explanation=EXPLANATION + "Test change: the expected value was wrong.\n",
    )
    body = build_pr_body(data(t))
    section = body.split("### Test changes", 1)[1].split("\n### ", 1)[0]
    assert "⚠ review" in body.split("### Test changes", 1)[1].splitlines()[0]
    assert "== 8 → assert subtract(5, 3) == 2" in section
    assert "Test change: the expected value was wrong." in section


def test_test_change_justification_not_in_fixed_excerpt_fallback() -> None:
    t = fixed(explanation="Test change: x was wrong\nplain line one\nplain line two\n")
    lines = explanation_excerpt(t.explanation)
    assert all(not ln.startswith("Test change") for ln in lines)


def test_source_change_note() -> None:
    body = build_pr_body(data(fixed(source_changed=True)))
    section = body.split("### Source changes", 1)[1].split("\n### ", 1)[0]
    assert T_SUB in section
    assert "review" in section.lower()


def test_no_source_section_for_test_only_fix() -> None:
    assert "Source changes" not in build_pr_body(data(fixed(source_changed=False)))


# ---- unfixable / not found / pre-existing / warnings ----------------------------------------


def test_unfixable_reason_truncated_and_attempts_shown() -> None:
    t = outcome(T_MEAN, OutcomeStatus.UNFIXABLE, reason="z" * 1000, attempts=3)
    body = build_pr_body(data(fixed(), t))
    section = body.split("### Unfixable", 1)[1].split("\n### ", 1)[0]
    line = next(ln for ln in section.splitlines() if T_MEAN in ln)
    assert "3 attempts" in line
    assert "z" * 300 not in line
    assert "z" * 250 in line


def test_unfixable_single_attempt_singular() -> None:
    t = outcome(T_MEAN, OutcomeStatus.UNFIXABLE, reason="r", attempts=1)
    assert "1 attempt)" in build_pr_body(data(fixed(), t))


def test_not_found_and_ambiguous() -> None:
    body = build_pr_body(
        data(
            fixed(),
            outcome("test_nope", OutcomeStatus.NOT_FOUND, node_id=None, reason="not collected"),
            outcome("test_twice", OutcomeStatus.AMBIGUOUS, node_id=None, reason="2 matches"),
        )
    )
    section = body.split("### Not found / ambiguous", 1)[1].split("\n### ", 1)[0]
    assert "test_nope" in section and "not collected" in section
    assert "test_twice" in section and "2 matches" in section


def test_preexisting_and_warnings() -> None:
    body = build_pr_body(
        data(
            fixed(),
            preexisting_failures=["tests/test_old.py::test_broken"],
            warnings=["regression check skipped: timeout"],
        )
    )
    assert "Pre-existing failures" in body
    assert "tests/test_old.py::test_broken" in body
    assert "### Warnings" in body
    assert "regression check skipped: timeout" in body


def test_long_preexisting_list_is_capped() -> None:
    ids = [f"tests/t.py::test_{i}" for i in range(100)]
    body = build_pr_body(data(fixed(), preexisting_failures=ids))
    assert "tests/t.py::test_99" not in body
    assert "more" in body


# ---- verification ---------------------------------------------------------------------------


@pytest.mark.parametrize("checked", [True, False])
@pytest.mark.parametrize("reviewer", [True, False])
def test_verification_variants(checked: bool, reviewer: bool) -> None:
    body = build_pr_body(data(fixed(), full_suite_checked=checked, reviewer_enabled=reviewer))
    section = body.split("### Verification", 1)[1].split("\n### ", 1)[0].split("---")[0]
    assert ("full test suite" in section.lower()) is checked
    assert ("reviewer" in section.lower()) is reviewer


# ---- comments -------------------------------------------------------------------------------


def test_fix_pr_comment_has_marker_link_and_statuses() -> None:
    comment = build_fix_pr_comment(
        FIX_PR_URL,
        data(fixed(T_SUB), outcome(T_MEAN, OutcomeStatus.UNFIXABLE, reason="hopeless", attempts=3)),
    )
    assert COMMENT_MARKER == "<!-- ci-fix -->"
    assert COMMENT_MARKER in comment
    assert FIX_PR_URL in comment
    assert T_SUB in comment and T_MEAN in comment
    assert "hopeless" in comment


def test_no_fix_comment_has_marker_and_reasons() -> None:
    comment = build_no_fix_comment(
        data(
            outcome(T_MEAN, OutcomeStatus.UNFIXABLE, reason="hopeless", attempts=3),
            outcome("test_nope", OutcomeStatus.NOT_FOUND, node_id=None, reason="not collected"),
        )
    )
    assert COMMENT_MARKER in comment
    assert "hopeless" in comment and "not collected" in comment
    assert "3 attempts" in comment


def test_body_and_commit_have_no_marker() -> None:
    assert COMMENT_MARKER not in build_commit_message(data(fixed()))


# ---- readability guard ----------------------------------------------------------------------


def test_typical_body_is_short() -> None:
    long_expl = (
        "I investigated the module carefully and ran the test several times.\n"
        "Root cause: the function had an off-by-one error in the loop bounds.\n"
        "Fix: use the correct upper bound and add a guard for empty input.\n"
        + "Extra reasoning line that should not be shown in the PR body.\n"
        * 20
    )
    tests = [
        fixed(T_SUB, explanation=long_expl, source_changed=True),
        fixed(T_DIV, explanation=long_expl, source_changed=True),
        outcome(T_MEAN, OutcomeStatus.UNFIXABLE, reason="r" * 2000, attempts=3),
    ]
    body = build_pr_body(data(*tests, warnings=["one warning"], full_suite_checked=True))
    assert len(body) < 2500, len(body)
    assert len(build_commit_message(data(*tests))) < 2500


# ---- sanitizing, caps, code spans (review fixes) --------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ping @octocat", f"ping @{ZWSP}octocat"),
        ("team @org/team-x", f"team @{ZWSP}org/team-x"),
        ("lonely @ sign", "lonely @ sign"),
        ("Fixes #12", f"Fixes {ZWSP}#12"),
        ("closes: #3", f"closes: {ZWSP}#3"),
        ("resolved owner/repo#9", f"resolved owner/repo{ZWSP}#9"),
        ("fix for #5", "fix for #5"),
        ("issue #5", "issue #5"),
        ("a <img src=x onerror=alert(1)>b", "a b"),
        ("<details><summary>s</summary>x</details>", "sx"),
        ("hidden <!-- note --> text", "hidden  text"),
        ("in <module> and <lambda>", "in <module> and <lambda>"),
    ],
)
def test_sanitize(raw: str, expected: str) -> None:
    assert sanitize(raw) == expected


def test_sanitized_fields_in_pr_body_and_commit() -> None:
    t = fixed(
        explanation="Root cause: @alice <b>broke</b> it\nFix: closes #4\nTest change: @bob said",
        test_changes=["tests/t.py::test_x: 1 → @carol"],
    )
    u = outcome("tests/t.py::test_u", OutcomeStatus.UNFIXABLE, reason="fixes #9 <script>x</script>")
    d = data(t, u, warnings=["see @dave"])
    for text in (build_pr_body(d), build_commit_message(d)):
        for name in ("alice", "bob", "carol", "dave"):
            assert f"@{ZWSP}{name}" in text and f"@{name}" not in text
        assert "<b>" not in text and "<script>" not in text
        assert f"closes {ZWSP}#4" in text and f"fixes {ZWSP}#9" in text


def test_comments_keep_marker_and_sanitize_reasons() -> None:
    u = outcome("tests/t.py::test_u", OutcomeStatus.UNFIXABLE, reason="cc @eve <!-- x -->")
    for text in (build_no_fix_comment(data(u)), build_fix_pr_comment("https://x/pull/1", data(u))):
        assert text.startswith(COMMENT_MARKER)
        assert f"@{ZWSP}eve" in text and "<!-- x" not in text


def test_own_strings_not_sanitized() -> None:
    body = build_pr_body(data(fixed()))
    assert ZWSP not in body  # "Fixes failing tests in #42", footer etc. stay as written
    assert body.startswith(f"Fixes failing tests in #{PR} ")


def test_cap_text_line_boundary() -> None:
    text = "".join(f"line {i}\n" for i in range(100))
    capped = cap_text(text, 100)
    assert len(capped) <= 100
    *kept, note = capped.rstrip("\n").split("\n")
    assert all(line.startswith("line ") for line in kept)
    assert note == f"… (truncated; {len(text) - sum(len(k) + 1 for k in kept)} more characters)"
    assert cap_text("short\n", 100) == "short\n"


def test_long_texts_capped() -> None:
    many = [
        outcome(f"tests/t.py::test_{i}", OutcomeStatus.UNFIXABLE, reason="r" * 300)
        for i in range(400)
    ]
    d = data(*many)
    for text in (
        build_pr_body(d),
        build_commit_message(d),
        build_no_fix_comment(d),
        build_fix_pr_comment("https://x/pull/1", d),
    ):
        assert len(text) <= TEXT_MAX
    assert "(truncated; " in build_pr_body(d)


def test_code_span_with_backtick() -> None:
    assert _code("a`b", True) == "`` a`b ``"
    assert _code("ab", True) == "`ab`"
    assert _code("a`b", False) == "a`b"
