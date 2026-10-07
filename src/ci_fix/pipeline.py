"""Public entry point: fix the failing tests of a pull request."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from ci_fix.agent import ClaudeFixer
from ci_fix.config import Settings, load_settings
from ci_fix.graph import PipelineDeps, PipelineState, RunContext, build_graph, recursion_limit
from ci_fix.guards.reviewer import ClaudeReviewer
from ci_fix.logging_setup import get_logger
from ci_fix.models import Fixer, FixResult, OutcomeStatus, TestOutcome
from ci_fix.tools.github import GitHubClient, parse_repo_url
from ci_fix.workspace import cleanup_workspace

log = get_logger(__name__)


def fix_failing_tests(
    repo_url: str,
    pr_number: int,
    failing_tests: Sequence[str],
    *,
    settings: Settings | None = None,
    fixer: Fixer | None = None,
    github: GitHubClient | None = None,
    deps: PipelineDeps | None = None,
) -> FixResult:
    """Check out PR ``pr_number`` of ``repo_url``, try to fix ``failing_tests``, report per test.

    If ``deps`` is given it is used as-is (``settings``/``fixer``/``github`` are ignored);
    it holds no per-run state, so one ``deps`` can serve concurrent runs.
    The workspace is always cleaned up (unless ``keep_workspace``), even when a step raises.
    """
    if isinstance(failing_tests, str):
        raise ValueError("failing_tests must be a sequence of test names, not a string")
    requested = list(dict.fromkeys(t.strip() for t in failing_tests if t and t.strip()))
    if not requested:
        raise ValueError("failing_tests must contain at least one test name")
    if pr_number <= 0:
        raise ValueError(f"pr_number must be positive, got {pr_number}")

    if deps is None:
        settings = settings if settings is not None else load_settings()
        if github is None:
            token = settings.github_token.get_secret_value() if settings.github_token else None
            github = GitHubClient(token)
        if fixer is None:
            fixer = ClaudeFixer(settings)  # raises ConfigError without ANTHROPIC_API_KEY
        reviewer = ClaudeReviewer(settings) if settings.review_test_changes else None
        deps = PipelineDeps(settings=settings, github=github, fixer=fixer, reviewer=reviewer)
    settings = deps.settings

    ref = parse_repo_url(repo_url)
    started = time.monotonic()
    log.info("Fixing %d test(s) on %s#%d", len(requested), ref.full_name, pr_number)

    graph = build_graph(deps)
    initial = PipelineState(repo_url=repo_url, pr_number=pr_number, requested=requested)
    ctx = RunContext()
    config = {
        "configurable": {"ctx": ctx},
        # ``requested`` is an upper bound on the number of unique test ids.
        "recursion_limit": recursion_limit(settings, len(requested)),
    }
    try:
        raw = graph.invoke(initial, config=config)
    finally:
        # If setup failed before ``prepared`` was set, prepare_pr_checkout cleaned up itself.
        if ctx.prepared is not None:
            cleanup_workspace(ctx.prepared, settings)
    final = _as_state(raw)

    tests = [
        final.outcomes.get(name)
        or TestOutcome(
            requested_name=name,
            node_id=final.name_to_id.get(name),
            status=OutcomeStatus.UNFIXABLE,
            reason="no outcome recorded",
        )
        for name in requested
    ]
    result = FixResult(
        repo_url=repo_url,
        pr_number=pr_number,
        branch=final.prepared.branch if final.prepared is not None else None,
        diff=final.diff,
        summary=final.summary,
        tests=tests,
        preexisting_failures=final.preexisting_failures,
        warnings=final.warnings,
    )
    counts = {s: sum(1 for t in tests if t.status == s) for s in OutcomeStatus}
    log.info(
        "Done in %.1fs: %s",
        time.monotonic() - started,
        ", ".join(f"{n} {s.value}" for s, n in counts.items() if n),
    )
    return result


def _as_state(raw: Any) -> PipelineState:
    if isinstance(raw, PipelineState):
        return raw
    return PipelineState.model_validate(dict(raw))
