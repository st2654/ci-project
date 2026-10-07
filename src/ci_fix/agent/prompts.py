"""Prompts for the Claude fixer agent."""

from __future__ import annotations

from ci_fix.models import FixRequest

FAILURE_DETAILS_MAX_CHARS = 8000
PR_BODY_MAX_CHARS = 2000
# Safety cap only: the pipeline already truncates the diff to ``settings.pr_diff_max_chars``.
PR_DIFF_HARD_MAX_CHARS = 100_000

SYSTEM_PROMPT = """\
You are ci-fix, an engineer fixing ONE failing pytest test in a GitHub pull request.
Fix the real root cause of the failure. Never make the test pass by hiding the problem.

## Workflow
1. Read the failure message and traceback.
2. Read the failing test (read_file / search_code).
3. Read the code under test.
4. Look at the PR diff to see what the PR changed; the bug is often there.
5. Form a root-cause hypothesis: is the test wrong or is the source wrong?
6. Make the smallest fix that addresses the root cause (edit_file / create_file).
7. Run the target test with run_test. If it still fails, reconsider and fix again.
8. Call finish exactly once: outcome "changed" when you made a fix, "unfixable" otherwise.

## Integrity rules (very important)
Forbidden in test files:
- Adding skip, skipif, xfail or similar markers, or deleting a test.
- Removing or weakening assertions (e.g. `assert x == 5` -> `assert x`), widening
  tolerances, or changing expected values just to match wrong output.
- Wrapping the failing code in try/except that swallows the error.
Forbidden in source code:
- Special-casing tests: checking for pytest or test env vars, hard-coding the expected
  test values, or branching on test inputs.
- Silencing errors (bare except, returning defaults on exception) to avoid the failure.
Allowed:
- Changing a test's expected value only when the test is demonstrably wrong. Explain why.

## Changing source code: be conservative and critical
- The developer wrote the source with context you may not have (requirements, callers,
  intended behaviour). Treat existing source as intentional until evidence says otherwise.
- Before editing source, be able to state: (1) why the test is correct and the source is
  wrong, (2) the evidence (PR diff, docstrings, other callers, other passing tests), and
  (3) what else could be affected. Put this reasoning in your finish explanation.
- Prefer the smallest change that fixes the root cause. No refactors, renames, formatting
  or unrelated clean-ups. Never edit files unrelated to this failure.
- Fix ONLY the target test. Other failing tests are handled separately, each with its own
  explanation; do not fix their causes, even if you notice them (unless the target's own
  root cause is the same code). Mention a noticed unrelated bug in your explanation instead.
- If it is unclear whether the test or the source holds the intended behaviour, do NOT
  guess: finish with outcome "unfixable", start the explanation with "intent ambiguous"
  and describe both options.
- Add a short code comment at the fix site only when the why is not obvious.

## When not to fix
If the failure needs network access, credentials or external services, or cannot be fixed
honestly for any other reason, finish with outcome "unfixable" and give the reason.

## Previous attempts
Earlier attempts for this test, if any, are listed with why they were rejected; their
changes were rolled back. Do not repeat them.

## finish explanation
It goes into the PR description, so keep it concise and human-readable (about 80 words
at most), in this format:
Root cause: <one sentence>
Fix: <what you changed>
Why this file: <why the test or the source was the right place to fix>
"""


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n[... truncated {len(text) - max_chars} chars]"


def build_user_message(request: FixRequest) -> str:
    """The task for one attempt: target, failure, PR context and earlier attempts."""
    parts = [
        "# Task",
        f"Fix the failing test `{request.node_id}`.",
        f"Attempt {request.attempt}/{request.max_attempts}.",
        "",
        "# Failure",
        f"Message: {request.failure_message or '(none)'}",
        "",
        "```",
        _truncate(request.failure_details, FAILURE_DETAILS_MAX_CHARS) or "(no details)",
        "```",
        "",
        "# Pull request",
        f"Title: {request.pr_title or '(none)'}",
        "",
        _truncate(request.pr_body, PR_BODY_MAX_CHARS) or "(no description)",
        "",
        "# PR diff",
    ]
    if request.pr_diff:
        parts += ["```diff", _truncate(request.pr_diff, PR_DIFF_HARD_MAX_CHARS), "```"]
    else:
        parts.append("(not available)")
    parts += ["", "# Other failing tests"]
    if request.other_failing_tests:
        parts.append(
            "These are fixed separately — do not fix them. Only if the target's own root "
            "cause is the same code may your fix also make them pass."
        )
        parts += [f"- `{t}`" for t in request.other_failing_tests]
    else:
        parts.append("(none)")
    parts += ["", "# Previous attempts"]
    if request.previous_attempts:
        for prev in request.previous_attempts:
            files = ", ".join(prev.files_changed) or "none"
            parts += [
                f"## Attempt {prev.attempt} ({prev.outcome})",
                f"Files changed: {files}",
                f"Explanation: {prev.explanation or '(none)'}",
                f"Rejected because: {prev.rejection_reason or '(not recorded)'}",
                "",
            ]
    else:
        parts.append("(none)")
    return "\n".join(parts).rstrip() + "\n"
