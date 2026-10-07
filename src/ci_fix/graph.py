"""LangGraph wiring for the fix pipeline.

setup → resolve → run_initial → select_next ⇄ fix_one → verify_one → … → finalize

Failing tests are fixed one at a time. Every attempt is verified against ALL requested
tests: an accepted attempt becomes a local checkpoint commit, a rejected one is rolled back
and its reason is passed to the next attempt.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from ci_fix.config import Settings
from ci_fix.logging_setup import get_logger
from ci_fix.models import FixAttempt, Fixer, FixRequest, OutcomeStatus, TestOutcome
from ci_fix.tools.git import GitRepo
from ci_fix.tools.github import GitHubClient
from ci_fix.tools.pytest_runner import (
    PytestRunner,
    TestResult,
    TestRunResult,
    TestStatus,
    resolve_test_names,
)
from ci_fix.tools.test_env import TestEnv, create_test_env
from ci_fix.workspace import PreparedRepo, prepare_pr_checkout

log = get_logger(__name__)

NOT_COLLECTED_REASON = "not collected by pytest"
DELETED_REASON = "test no longer collected after fix — fixes must not delete or rename tests"
SKIPPED_AFTER_FIX_REASON = "test skipped after fix — fixes must not skip tests"
NO_CHANGES_REASON = "no changes made"
_FAILING = (TestStatus.FAILED, TestStatus.ERROR)


class TestRunner(Protocol):
    """What the graph needs from a test runner (``PytestRunner`` or a fake in tests)."""

    __test__ = False

    collection_errors: set[str]

    def collect(self) -> list[str]: ...

    def run(self, node_ids: Sequence[str]) -> TestRunResult: ...


def _default_runner_factory(
    repo_path: Path, python: Path, reports_dir: Path, settings: Settings
) -> TestRunner:
    return PytestRunner(
        repo_path, python, reports_dir, settings.pytest_args, settings.test_timeout_seconds
    )


@dataclass(frozen=True)
class PipelineDeps:
    """Collaborators the graph nodes use; swap any of them in tests. Shared between runs."""

    settings: Settings
    github: GitHubClient
    fixer: Fixer
    env_factory: Callable[[Path, Path, Settings], TestEnv] = create_test_env
    runner_factory: Callable[[Path, Path, Path, Settings], TestRunner] = _default_runner_factory
    clone_url: str | None = None


@dataclass
class RunContext:
    """Per-run handles, filled in by the setup nodes; one per ``fix_failing_tests`` call.

    The caller keeps a reference so it can clean up the workspace even when a node raises.
    """

    prepared: PreparedRepo | None = None
    runner: TestRunner | None = None

    @property
    def repo(self) -> GitRepo:
        if self.prepared is None:
            raise RuntimeError("workspace not prepared (setup_repo did not run)")
        return GitRepo(self.prepared.path)


class PipelineState(BaseModel):
    """LangGraph state. Nodes return partial updates with full replacement values."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    repo_url: str
    pr_number: int
    requested: list[str]
    prepared: PreparedRepo | None = None
    python: Path | None = None
    name_to_id: dict[str, str] = Field(default_factory=dict)
    outcomes: dict[str, TestOutcome] = Field(default_factory=dict)  # keyed by requested name
    pending: list[str] = Field(default_factory=list)  # node ids still failing, in order
    current: str | None = None  # node id being fixed
    attempts: dict[str, int] = Field(default_factory=dict)
    history: dict[str, list[FixAttempt]] = Field(default_factory=dict)
    # Results of all resolved ids at the current HEAD (the last accepted state).
    results: dict[str, TestResult] = Field(default_factory=dict)
    diff: str = ""
    summary: str = ""


def recursion_limit(settings: Settings, n_tests: int) -> int:
    """Graph step limit for ``n_tests`` unique test ids.

    Each attempt is at most 3 steps (select_next → fix_one → verify_one); 20 covers the
    fixed steps (setup, resolve, run, final select_next, finalize) with room to spare.
    """
    return n_tests * settings.max_attempts * 3 + 20


def _ctx(config: RunnableConfig) -> RunContext:
    return config["configurable"]["ctx"]


def _runner(config: RunnableConfig) -> TestRunner:
    runner = _ctx(config).runner
    if runner is None:
        raise RuntimeError("test runner not initialised (setup_env did not run)")
    return runner


def _unique_ids(state: PipelineState) -> list[str]:
    return list(dict.fromkeys(state.name_to_id.values()))


def _names_for(state: PipelineState, node_id: str) -> list[str]:
    return [name for name, nid in state.name_to_id.items() if nid == node_id]


def _not_found(node_id: str) -> TestResult:
    return TestResult(node_id=node_id, status=TestStatus.NOT_FOUND, message=NOT_COLLECTED_REASON)


def _set_outcome(
    outcomes: dict[str, TestOutcome],
    state: PipelineState,
    node_id: str,
    status: OutcomeStatus,
    reason: str = "",
    attempts: int = 0,
    files_changed: list[str] | None = None,
) -> None:
    for name in _names_for(state, node_id):
        outcomes[name] = TestOutcome(
            requested_name=name,
            node_id=node_id,
            status=status,
            reason=reason,
            attempts=attempts,
            files_changed=list(files_changed or []),
        )
    if status == OutcomeStatus.UNFIXABLE:
        log.warning("[unfixable] %s: %s", node_id, reason)


@dataclass
class Verdict:
    """How the results after an attempt compare with the results before it."""

    target: str = ""  # why the target is not fixed ("" = it passes)
    broken: list[str] = field(default_factory=list)  # passed (or skipped) before, fail now
    vanished: list[str] = field(default_factory=list)  # no longer collected
    skipped: list[str] = field(default_factory=list)  # newly skipped

    @property
    def only_broken(self) -> bool:
        return bool(self.broken) and not (self.target or self.vanished or self.skipped)

    def reasons(self) -> list[str]:
        """Why the attempt must be rejected ([] = accept)."""
        reasons = [self.target] if self.target else []
        if self.broken:
            reasons.append(f"broke previously passing tests: {', '.join(self.broken)}")
        if self.vanished:
            reasons.append(
                f"tests no longer collected after fix: {', '.join(self.vanished)}"
                " — fixes must not delete or rename tests"
            )
        if self.skipped:
            reasons.append(
                f"tests skipped after fix: {', '.join(self.skipped)} — fixes must not skip tests"
            )
        return reasons


def judge(target: str, before: dict[str, TestResult], after: dict[str, TestResult]) -> Verdict:
    """Compare the results ``after`` an attempt for ``target`` with those ``before`` it."""
    verdict = Verdict()
    res = after[target]
    if res.status == TestStatus.NOT_FOUND:
        verdict.target = DELETED_REASON
    elif res.status == TestStatus.SKIPPED:
        verdict.target = SKIPPED_AFTER_FIX_REASON
    elif res.status != TestStatus.PASSED:
        verdict.target = (
            f"target still failing: {res.message}" if res.message else "target still failing"
        )
    for nid, new in after.items():
        old = before.get(nid)
        if nid == target or old is None:
            continue
        if old.status in (TestStatus.PASSED, TestStatus.SKIPPED) and new.status in _FAILING:
            verdict.broken.append(nid)
        elif new.status == TestStatus.NOT_FOUND and old.status != TestStatus.NOT_FOUND:
            verdict.vanished.append(nid)
        elif new.status == TestStatus.SKIPPED and old.status != TestStatus.SKIPPED:
            verdict.skipped.append(nid)
    return verdict


def build_summary(state: PipelineState) -> str:
    """Placeholder markdown summary: one line per requested test."""
    lines = [f"## ci-fix results for PR #{state.pr_number}", ""]
    for name in state.requested:
        outcome = state.outcomes.get(name)
        if outcome is None:
            lines.append(f"- `{name}`: no outcome recorded")
            continue
        line = f"- `{name}`: **{outcome.status.value}**"
        if outcome.reason:
            line += f" — {outcome.reason}"
        lines.append(line)
    return "\n".join(lines) + "\n"


def build_graph(deps: PipelineDeps) -> Any:
    """Build and compile the pipeline graph. Per-run handles come from ``config["ctx"]``."""
    settings = deps.settings

    def setup_repo(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        log.info("[setup] Preparing %s PR #%d", state.repo_url, state.pr_number)
        prepared = prepare_pr_checkout(
            state.repo_url, state.pr_number, settings, deps.github, deps.clone_url
        )
        _ctx(config).prepared = prepared
        return {"prepared": prepared}

    def setup_env(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        assert state.prepared is not None
        prepared = state.prepared
        log.info("[env] Creating test environment")
        env = deps.env_factory(prepared.path, prepared.venv_dir, settings)
        # Build artifacts (``*.egg-info``, …) must never end up in a fix.
        _ctx(config).repo.exclude_untracked()
        _ctx(config).runner = deps.runner_factory(
            prepared.path, env.python, prepared.reports_dir, settings
        )
        return {"python": env.python}

    def resolve_tests(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        log.info("[resolve] Resolving %d test name(s)", len(state.requested))
        r = _runner(config)
        collected = r.collect()
        name_to_id: dict[str, str] = {}
        outcomes = dict(state.outcomes)
        for name in state.requested:
            # A file that fails to import has no collected tests, but its full ids are still
            # valid targets: the import error may be the bug to fix.
            if "::" in name and name.split("::", 1)[0] in r.collection_errors:
                name_to_id[name] = name
                continue
            resolved = resolve_test_names([name], collected)
            if resolved.node_ids:
                name_to_id[name] = resolved.node_ids[0]
            elif name in resolved.ambiguous:
                candidates = resolved.ambiguous[name]
                outcomes[name] = TestOutcome(
                    requested_name=name,
                    node_id=None,
                    status=OutcomeStatus.AMBIGUOUS,
                    reason=f"ambiguous name; candidates: {', '.join(candidates)}",
                )
            else:
                outcomes[name] = TestOutcome(
                    requested_name=name,
                    node_id=None,
                    status=OutcomeStatus.NOT_FOUND,
                    reason=NOT_COLLECTED_REASON,
                )
        log.info(
            "[resolve] %d resolved, %d not resolved",
            len(name_to_id),
            len(state.requested) - len(name_to_id),
        )
        return {"name_to_id": name_to_id, "outcomes": outcomes}

    def run_tests(ids: list[str], config: RunnableConfig) -> dict[str, TestResult]:
        run = _runner(config).run(ids)
        return {nid: run.results.get(nid) or _not_found(nid) for nid in ids}

    def run_initial(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        log.info("[run] Running %d test(s) to confirm failures", len(_unique_ids(state)))
        results = run_tests(_unique_ids(state), config)
        # Whatever the env setup and this run left behind (``.coverage``, …) is not a fix.
        _ctx(config).repo.exclude_untracked()
        outcomes = dict(state.outcomes)
        pending: list[str] = []
        for nid, res in results.items():
            if res.status == TestStatus.NOT_FOUND:
                reason = res.message or NOT_COLLECTED_REASON
                _set_outcome(outcomes, state, nid, OutcomeStatus.NOT_FOUND, reason)
            elif res.status == TestStatus.PASSED:
                _set_outcome(outcomes, state, nid, OutcomeStatus.ALREADY_PASSING)
            elif res.status == TestStatus.SKIPPED:
                _set_outcome(
                    outcomes, state, nid, OutcomeStatus.ALREADY_PASSING, "skipped by pytest"
                )
            else:
                pending.append(nid)
        log.info("[run] %d failing test(s) to fix", len(pending))
        return {"outcomes": outcomes, "pending": pending, "results": results}

    def select_next(state: PipelineState) -> dict[str, Any]:
        # Tests fixed as a side effect of another fix are removed from ``pending`` by
        # verify_one, so the first pending id always still fails at HEAD.
        current = state.pending[0] if state.pending else None
        if current is not None:
            log.debug("[fix] Next test: %s (%d pending)", current, len(state.pending))
        return {"current": current}

    def reject(
        state: PipelineState,
        previous: list[FixAttempt],
        attempt: FixAttempt,
        reason: str,
        update: dict[str, Any],
    ) -> dict[str, Any]:
        """Record a rejected attempt; give up on the test once attempts run out."""
        nid, n = attempt.node_id, attempt.attempt
        attempt = attempt.model_copy(update={"accepted": False, "rejection_reason": reason})
        update = {**update, "history": {**state.history, nid: [*previous, attempt]}}
        if n < settings.max_attempts:
            log.info(
                "[verify] %s attempt %d/%d rejected: %s", nid, n, settings.max_attempts, reason
            )
            return update
        outcomes = dict(state.outcomes)
        reason = f"still failing after {n} attempt(s): {reason}"
        _set_outcome(outcomes, state, nid, OutcomeStatus.UNFIXABLE, reason, n)
        pending = [p for p in state.pending if p != nid]
        return {**update, "outcomes": outcomes, "pending": pending}

    def fix_one(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        assert state.prepared is not None and state.current is not None
        nid = state.current
        repo = _ctx(config).repo
        n = state.attempts.get(nid, 0) + 1
        failure = state.results.get(nid)
        previous = state.history.get(nid, [])
        log.info("[fix] %s (attempt %d/%d)", nid, n, settings.max_attempts)
        request = FixRequest(
            node_id=nid,
            repo_path=state.prepared.path,
            attempt=n,
            max_attempts=settings.max_attempts,
            failure_message=failure.message if failure else "",
            failure_details=failure.details if failure else "",
            previous_attempts=list(previous),
        )
        started = time.monotonic()
        update: dict[str, Any] = {"attempts": {**state.attempts, nid: n}}
        try:
            attempt = deps.fixer.fix(request)
        except Exception as exc:  # a fixer bug must not crash the whole run
            log.error("[fix] fixer raised for %s: %s", nid, exc, exc_info=True)
            repo.rollback()  # drop any partial edits
            failed = FixAttempt(
                node_id=nid, attempt=n, outcome="no_change", explanation=f"fixer error: {exc}"
            )
            return reject(state, previous, failed, f"fixer error: {exc}", update)

        files = repo.changed_files()
        if sorted(set(attempt.files_changed)) != files:
            log.debug("[fix] fixer reported %s, git shows %s", attempt.files_changed, files)
        attempt = attempt.model_copy(update={"node_id": nid, "attempt": n, "files_changed": files})
        log.info(
            "[fix] %s: %s in %.1fs%s",
            nid,
            attempt.outcome,
            time.monotonic() - started,
            f" ({', '.join(files)})" if files else "",
        )
        if attempt.outcome == "unfixable":
            repo.rollback()
            outcomes = dict(state.outcomes)
            _set_outcome(outcomes, state, nid, OutcomeStatus.UNFIXABLE, attempt.explanation, n)
            history = {**state.history, nid: [*previous, attempt]}
            pending = [p for p in state.pending if p != nid]
            return {**update, "outcomes": outcomes, "pending": pending, "history": history}
        if not files:
            return reject(state, previous, attempt, NO_CHANGES_REASON, update)
        history = {**state.history, nid: [*previous, attempt]}
        return {**update, "history": history}

    def verify_one(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        assert state.current is not None
        nid = state.current
        repo = _ctx(config).repo
        n = state.attempts[nid]
        *earlier, attempt = state.history[nid]
        log.info("[verify] Re-running %d test(s)", len(_unique_ids(state)))
        untracked_before = repo.untracked_files()
        after = run_tests(_unique_ids(state), config)
        verdict = judge(nid, state.results, after)
        if verdict.only_broken:
            # Re-run once so a flaky test doesn't sink a good fix.
            after = {**after, **run_tests(verdict.broken, config)}
            for other in verdict.broken:
                if after[other].status == TestStatus.PASSED:
                    log.info("[verify] %s passed on re-run; treating as flaky", other)
            verdict = judge(nid, state.results, after)
        reasons = verdict.reasons()
        if not reasons:
            # Confirm side-effect fixes with a re-run before counting them as fixed.
            side = [o for o in state.pending if o != nid and after[o].status == TestStatus.PASSED]
            if side:
                after = {**after, **run_tests(side, config)}
                for other in side:
                    if after[other].status != TestStatus.PASSED:
                        log.info("[verify] %s failed on re-run; treating as flaky", other)
        # Files the test runs created (not the fixer's) are artifacts: keep them out.
        repo.exclude(repo.untracked_files() - untracked_before)
        if reasons:
            repo.rollback()  # HEAD (and so ``state.results``) is the pre-attempt state
            return reject(state, earlier, attempt, "; ".join(reasons), {})

        repo.checkpoint(f"ci-fix: fix {nid} (attempt {n})")
        files = attempt.files_changed
        log.info("[verify] %s fixed after %d attempt(s)", nid, n)
        outcomes = dict(state.outcomes)
        _set_outcome(outcomes, state, nid, OutcomeStatus.FIXED, "", n, files)
        pending = []
        for other in state.pending:
            if other == nid:
                continue
            if after[other].status == TestStatus.PASSED:
                log.info("[verify] %s also fixed by the fix for %s", other, nid)
                reason = f"fixed by the fix for {nid}"
                attempts = state.attempts.get(other, 0)
                _set_outcome(outcomes, state, other, OutcomeStatus.FIXED, reason, attempts, files)
            else:
                pending.append(other)
        accepted = attempt.model_copy(update={"accepted": True})
        return {
            "outcomes": outcomes,
            "pending": pending,
            "results": after,
            "history": {**state.history, nid: [*earlier, accepted]},
        }

    def finalize(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        diff = ""
        if state.prepared is not None:
            # The worktree is clean: the diff is exactly the accepted checkpoint commits.
            diff = _ctx(config).repo.diff(state.prepared.pr_head_sha)
        summary = build_summary(state)
        counts: dict[str, int] = {}
        for outcome in state.outcomes.values():
            counts[outcome.status.value] = counts.get(outcome.status.value, 0) + 1
        log.info(
            "[finalize] %s; diff %d line(s)",
            ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "no outcomes",
            len(diff.splitlines()),
        )
        return {"diff": diff, "summary": summary}

    def after_resolve(state: PipelineState) -> str:
        return "run_initial" if state.name_to_id else "finalize"

    def after_select(state: PipelineState) -> str:
        return "fix_one" if state.current is not None else "finalize"

    def after_fix(state: PipelineState) -> str:
        # Only an attempt that changed files and is still unjudged goes to verification.
        history = state.history.get(state.current or "", [])
        if history and history[-1].accepted is None and history[-1].outcome != "unfixable":
            return "verify_one"
        return "select_next"

    graph = StateGraph(PipelineState)
    graph.add_node("setup_repo", setup_repo)
    graph.add_node("setup_env", setup_env)
    graph.add_node("resolve_tests", resolve_tests)
    graph.add_node("run_initial", run_initial)
    graph.add_node("select_next", select_next)
    graph.add_node("fix_one", fix_one)
    graph.add_node("verify_one", verify_one)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "setup_repo")
    graph.add_edge("setup_repo", "setup_env")
    graph.add_edge("setup_env", "resolve_tests")
    graph.add_conditional_edges("resolve_tests", after_resolve, ["run_initial", "finalize"])
    graph.add_edge("run_initial", "select_next")
    graph.add_conditional_edges("select_next", after_select, ["fix_one", "finalize"])
    graph.add_conditional_edges("fix_one", after_fix, ["verify_one", "select_next"])
    graph.add_edge("verify_one", "select_next")
    graph.add_edge("finalize", END)
    return graph.compile()
