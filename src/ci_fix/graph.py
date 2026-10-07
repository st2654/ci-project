"""LangGraph wiring for the fix pipeline.

setup → resolve → run_initial → select_next ⇄ fix_one → verify_one → … → finalize

Failing tests are fixed one at a time. Every attempt is verified against ALL requested
tests: an accepted attempt becomes a local checkpoint commit, a rejected one is rolled back
and its reason is passed to the next attempt. An attempt that changed source (non-test) files
must also not break any test of the full suite that passed before (regression check).

Parallel fixing (``max_parallel_workers > 1``): when the pending tests fall into several
independent groups (see :mod:`ci_fix.triage`), a round runs one fixer per group in its own git
worktree (``plan_round`` → ``fix_in_worktree`` via ``Send``); ``merge_candidates`` then applies
each candidate patch to the main checkout and judges it serially, exactly like a sequential
attempt. A patch that no longer applies is retried sequentially.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Protocol

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, ConfigDict, Field

from ci_fix.config import Settings
from ci_fix.guards.patch_checker import CheckContext, PatchReport, check_patch
from ci_fix.guards.reviewer import TestChangeReviewer
from ci_fix.logging_setup import get_logger
from ci_fix.models import (
    FixAttempt,
    Fixer,
    FixerFatalError,
    FixRequest,
    OutcomeStatus,
    TestOutcome,
)
from ci_fix.tools.git import GitError, GitRepo
from ci_fix.tools.github import GitHubClient
from ci_fix.tools.pytest_runner import (
    PytestRunner,
    TestResult,
    TestRunError,
    TestRunResult,
    TestStatus,
    resolve_test_names,
)
from ci_fix.tools.target_env import build_target_env
from ci_fix.tools.test_env import TestEnv, create_test_env
from ci_fix.triage import group_failures
from ci_fix.workspace import PreparedRepo, prepare_pr_checkout

log = get_logger(__name__)

NOT_COLLECTED_REASON = "not collected by pytest"
DELETED_REASON = "test no longer collected after fix — fixes must not delete or rename tests"
SKIPPED_AFTER_FIX_REASON = "test skipped after fix — fixes must not skip tests"
NO_CHANGES_REASON = "no changes made"
ONLY_TARGET_REASON = "only the target test can be run"
INTEGRITY_PREFIX = "integrity check failed:"
REVIEWER_PREFIX = "reviewer rejected test change:"
REVIEWER_UNAVAILABLE_PREFIX = "reviewer unavailable:"
# A reviewer call that errors (not fatally) is retried; after this many tries the attempt is
# rejected like any other (it counts as an attempt): rejecting when in doubt is the safe side.
REVIEW_TRIES = 2
REGRESSION_PREFIX = "broke other tests in the full suite:"
REGRESSION_SKIPPED_PREFIX = "regression check skipped:"
RESTORE_FAILED_PREFIX = "could not restore the attempt after the baseline run:"
REGRESSION_LIST_MAX = 10  # regressed ids named in a rejection reason
_FAILING = (TestStatus.FAILED, TestStatus.ERROR)


class TestRunner(Protocol):
    """What the graph needs from a test runner (``PytestRunner`` or a fake in tests)."""

    __test__ = False

    collection_errors: set[str]

    def collect(self) -> list[str]: ...

    def run(
        self, node_ids: Sequence[str], extra_args: Sequence[str] = (), timeout: float | None = None
    ) -> TestRunResult: ...

    def run_all(
        self, extra_args: Sequence[str] = (), timeout: float | None = None
    ) -> TestRunResult: ...


def _default_runner_factory(
    repo_path: Path, python: Path, reports_dir: Path, settings: Settings
) -> TestRunner:
    return PytestRunner(
        repo_path, python, reports_dir, settings.pytest_args, settings.test_timeout_seconds
    )


def worktree_pythonpath(worktree: Path) -> list[Path]:
    """``PYTHONPATH`` entries that make a worktree's code shadow the editable install.

    The test venv's editable install points at the main checkout, so tests run in a worktree
    would import the main checkout's code. ``<worktree>/src`` (if present) and the worktree
    root come first instead. *Limitation:* other layouts (packages in other dirs, compiled
    extensions, ``package_dir`` mappings) may still import the main checkout's code.
    """
    worktree = Path(worktree)
    src = worktree / "src"
    return [src, worktree] if src.is_dir() else [worktree]


def _default_worktree_runner_factory(
    repo_path: Path, python: Path, reports_dir: Path, settings: Settings
) -> TestRunner:
    env = build_target_env(Path(python).parent.parent, worktree_pythonpath(repo_path))
    return PytestRunner(
        repo_path,
        python,
        reports_dir,
        settings.pytest_args,
        settings.test_timeout_seconds,
        env=env,
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
    # Optional second opinion on test-file changes (see ``Settings.review_test_changes``).
    reviewer: TestChangeReviewer | None = None
    # Builds the runner for a parallel-fix worktree. None = a ``PytestRunner`` whose
    # ``PYTHONPATH`` puts the worktree's code first when ``runner_factory`` is the default,
    # otherwise ``runner_factory`` itself (e.g. a fake runner in tests).
    worktree_runner_factory: Callable[[Path, Path, Path, Settings], TestRunner] | None = None

    def make_worktree_runner(self, repo_path: Path, python: Path, reports_dir: Path) -> TestRunner:
        factory = self.worktree_runner_factory
        if factory is None:
            factory = (
                _default_worktree_runner_factory
                if self.runner_factory is _default_runner_factory
                else self.runner_factory
            )
        return factory(repo_path, python, reports_dir, self.settings)


@dataclass
class RunContext:
    """Per-run handles, filled in by the setup nodes; one per ``fix_failing_tests`` call.

    The caller keeps a reference so it can clean up the workspace even when a node raises.
    """

    prepared: PreparedRepo | None = None
    runner: TestRunner | None = None
    pr_diff: str | None = None  # cached, truncated PR diff for the fixer

    @property
    def repo(self) -> GitRepo:
        if self.prepared is None:
            raise RuntimeError("workspace not prepared (setup_repo did not run)")
        return GitRepo(self.prepared.path)


class Candidate(BaseModel):
    """What one parallel fixer produced in its worktree, awaiting the serial merge."""

    node_id: str
    attempt: FixAttempt | None = None  # None when the fixer raised
    patch: bytes = b""  # raw binary git diff of the fixer's changes against the round's HEAD
    error: str | None = None  # fixer (or worktree) error message
    elapsed: float = 0.0  # seconds the fixer took


def _collect_candidates(old: list[Candidate], new: list[Candidate]) -> list[Candidate]:
    """Reducer: parallel branches append their candidates; an empty update clears the list."""
    return [*old, *new] if new else []


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
    # Patch-checker report of the attempt awaiting verification.
    patch_report: PatchReport | None = None
    # Full-suite statuses at HEAD, set at the first attempt that changes source files and
    # replaced after each accepted source change. None = not computed yet.
    regression_baseline: dict[str, TestStatus] | None = None
    # The baseline full-suite run failed (e.g. timeout): no regression checks this run.
    regression_unavailable: bool = False
    preexisting_failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # Parallel fixing: rounds started so far, candidates of the current round (appended by
    # the parallel branches, cleared by the merge) and tests whose patch conflicted with an
    # accepted fix, to be retried sequentially before the next round.
    parallel_round: int = 0
    candidates: Annotated[list[Candidate], _collect_candidates] = Field(default_factory=list)
    serial_queue: list[str] = Field(default_factory=list)
    diff: str = ""
    summary: str = ""


def recursion_limit(settings: Settings, n_tests: int) -> int:
    """Graph step limit for ``n_tests`` unique test ids.

    Sequentially each attempt is at most 3 steps (select_next → fix_one → verify_one). A
    parallel round is 4 steps (select_next → plan_round → fix_in_worktree → merge_candidates).
    Every candidate either uses up an attempt or, if its patch does not apply, is queued for a
    sequential retry (3 steps) that does. Worst case a round with one candidate that does not
    apply: 4 + 3 = 7 steps for one attempt, so 7 per attempt bounds parallel runs. 20 covers
    the fixed steps (setup, resolve, run, final select_next, finalize).
    """
    per_attempt = 7 if settings.max_parallel_workers > 1 else 3
    return n_tests * settings.max_attempts * per_attempt + 20


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
    **details: Any,
) -> None:
    """Set the outcome of every requested name for ``node_id``.

    ``details`` are extra TestOutcome fields (``explanation``, ``test_changes``, …).
    """
    for name in _names_for(state, node_id):
        outcomes[name] = TestOutcome(
            requested_name=name,
            node_id=node_id,
            status=status,
            reason=reason,
            attempts=attempts,
            files_changed=list(files_changed or []),
            **details,
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


def _covers(requested_id: str, node_id: str) -> bool:
    """Whether ``node_id`` is ``requested_id`` itself or one of its parametrized cases."""
    return node_id == requested_id or node_id.startswith(requested_id + "[")


def find_regressions(
    baseline: dict[str, TestStatus], current: dict[str, TestStatus], requested: Sequence[str]
) -> list[str]:
    """Ids that passed in ``baseline`` and now fail, error or are missing (sorted).

    Requested tests are left out: their verdict comes from :func:`judge`.
    """
    regressed = []
    for nid, status in baseline.items():
        if status != TestStatus.PASSED or any(_covers(r, nid) for r in requested):
            continue
        now = current.get(nid)
        if now is None or now in _FAILING or now == TestStatus.NOT_FOUND:
            regressed.append(nid)
    return sorted(regressed)


def _id_list(ids: Sequence[str]) -> str:
    """Up to ``REGRESSION_LIST_MAX`` ids, then "and N more"."""
    shown = ", ".join(ids[:REGRESSION_LIST_MAX])
    extra = len(ids) - REGRESSION_LIST_MAX
    return shown + (f" and {extra} more" if extra > 0 else "")


def regression_reason(failing: Sequence[str], missing: Sequence[str] = ()) -> str:
    """Rejection reason: ids now failing, then ids no longer collected (each list capped)."""
    parts = [_id_list(failing)] if failing else []
    if missing:
        parts.append(f"no longer collected: {_id_list(missing)}")
    return f"{REGRESSION_PREFIX} {'; '.join(parts)}"


def preexisting_failures(baseline: dict[str, TestStatus], requested: Sequence[str]) -> list[str]:
    """Ids failing in ``baseline`` that are not requested tests (sorted)."""
    return sorted(
        nid
        for nid, status in baseline.items()
        if status in _FAILING and not any(_covers(r, nid) for r in requested)
    )


def truncate_text(text: str, max_chars: int) -> str:
    """``text`` cut to at most ``max_chars`` (marker included) if it was longer.

    The cut part is replaced by a ``[... truncated N chars]`` marker.
    """
    if len(text) <= max_chars:
        return text
    # The marker for the largest possible N is never shorter than the real one.
    keep = max(0, max_chars - len(f"\n[... truncated {len(text)} chars]"))
    return f"{text[:keep]}\n[... truncated {len(text) - keep} chars]"


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

    def run_tests(
        ids: list[str], config: RunnableConfig, runner: TestRunner | None = None
    ) -> dict[str, TestResult]:
        run = (runner or _runner(config)).run(ids)
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

    def groups_of(state: PipelineState) -> list[list[str]]:
        """Independent groups of the pending tests, from their failure tracebacks at HEAD."""
        assert state.prepared is not None
        details = {nid: res.details for nid, res in state.results.items()}
        return group_failures(state.pending, details, state.prepared.path)

    def select_next(state: PipelineState) -> dict[str, Any]:
        # Tests fixed as a side effect of another fix are removed from ``pending`` by
        # verify_one, so every pending id still fails at HEAD.
        if settings.max_parallel_workers > 1 and state.pending:
            queue = [q for q in state.serial_queue if q in state.pending]
            if queue:  # a patch that conflicted in the last parallel round: retry it alone
                log.debug("[fix] Sequential retry: %s", queue[0])
                return {"current": queue[0], "serial_queue": queue[1:]}
            if len(groups_of(state)) > 1:
                return {"current": None, "serial_queue": []}  # → plan_round
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

    def pr_diff(prepared: PreparedRepo, ctx: RunContext) -> str:
        """What the PR changed (merge-base..PR head), truncated; computed once per run."""
        if ctx.pr_diff is None:
            diff = ""
            if prepared.base_sha:
                try:
                    diff = ctx.repo.diff_commits(prepared.base_sha, prepared.pr_head_sha)
                except GitError as exc:
                    log.warning("[fix] Could not compute the PR diff: %s", exc)
            ctx.pr_diff = truncate_text(diff, settings.pr_diff_max_chars)
        return ctx.pr_diff

    def make_run_test(
        target: str,
        config: RunnableConfig,
        repo: GitRepo | None = None,
        runner: TestRunner | None = None,
        on_artifacts: Callable[[set[str]], None] | None = None,
    ) -> Callable[[str], TestResult]:
        """``run_test`` for the fixer: runs ``target`` only, in ``repo`` with ``runner``
        (default: the main checkout and its runner).

        Files a run creates are artifacts: by default they go to ``info/exclude``; a
        worktree passes ``on_artifacts`` to collect them privately instead (the exclude file
        is shared by all worktrees and could hide another branch's new file).
        """
        repo = repo or _ctx(config).repo
        record = on_artifacts or repo.exclude

        def run_test(node_id: str) -> TestResult:
            if node_id != target:
                return TestResult(
                    node_id=node_id, status=TestStatus.NOT_FOUND, message=ONLY_TARGET_REASON
                )
            untracked_before = repo.untracked_files()
            try:
                return run_tests([node_id], config, runner)[node_id]
            finally:
                # Files the test run created are artifacts, not fixer changes.
                record(repo.untracked_files() - untracked_before)

        return run_test

    def make_request(
        state: PipelineState,
        nid: str,
        n: int,
        previous: list[FixAttempt],
        config: RunnableConfig,
        repo_path: Path | None = None,
        run_test: Callable[[str], TestResult] | None = None,
    ) -> FixRequest:
        """The fixer's (and reviewer's) view of attempt ``n`` for ``nid`` at the current HEAD.

        ``repo_path``/``run_test`` default to the main checkout (a worktree when parallel).
        """
        assert state.prepared is not None
        failure = state.results.get(nid)
        return FixRequest(
            node_id=nid,
            repo_path=repo_path or state.prepared.path,
            attempt=n,
            max_attempts=settings.max_attempts,
            failure_message=failure.message if failure else "",
            failure_details=failure.details if failure else "",
            previous_attempts=list(previous),
            pr_title=state.prepared.pr.title,
            pr_body=state.prepared.pr.body,
            pr_diff=pr_diff(state.prepared, _ctx(config)),
            other_failing_tests=[p for p in state.pending if p != nid],
            run_test=run_test or make_run_test(nid, config),
        )

    def run_check(repo: GitRepo, nid: str, explanation: str) -> tuple[PatchReport | None, str]:
        """Run the patch checker; a checker crash gives ``(None, "<Type>: <msg>")``."""
        test_path = nid.split("::", 1)[0]
        source = repo.file_at("HEAD", test_path)
        context = CheckContext(test_files={test_path: source} if source is not None else {})
        try:
            return check_patch(repo, explanation, context), ""
        except Exception as exc:  # a checker bug must neither crash the run nor pass a patch
            log.error("[integrity] checker error for %s: %s: %s", nid, type(exc).__name__, exc)
            log.debug("[integrity] checker traceback", exc_info=True)
            return None, f"{type(exc).__name__}: {exc}"

    def fix_one(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        assert state.prepared is not None and state.current is not None
        nid = state.current
        repo = _ctx(config).repo
        n = state.attempts.get(nid, 0) + 1
        previous = state.history.get(nid, [])
        log.info("[fix] %s (attempt %d/%d)", nid, n, settings.max_attempts)
        request = make_request(state, nid, n, previous, config)
        started = time.monotonic()
        update: dict[str, Any] = {"attempts": {**state.attempts, nid: n}}
        try:
            attempt = deps.fixer.fix(request)
        except FixerFatalError as exc:
            repo.rollback()
            log.error("[fix] stopping the run: %s", exc)
            raise
        except Exception as exc:  # a fixer bug must not crash the whole run
            log.error("[fix] fixer raised for %s: %s", nid, exc, exc_info=True)
            repo.rollback()  # drop any partial edits
            return reject_fixer_error(state, nid, n, previous, str(exc), update)
        elapsed = time.monotonic() - started
        return after_fixer(state, nid, n, previous, attempt, update, config, elapsed)

    def reject_fixer_error(
        state: PipelineState,
        nid: str,
        n: int,
        previous: list[FixAttempt],
        error: str,
        update: dict[str, Any],
    ) -> dict[str, Any]:
        failed = FixAttempt(
            node_id=nid, attempt=n, outcome="no_change", explanation=f"fixer error: {error}"
        )
        return reject(state, previous, failed, f"fixer error: {error}", update)

    def after_fixer(
        state: PipelineState,
        nid: str,
        n: int,
        previous: list[FixAttempt],
        attempt: FixAttempt,
        update: dict[str, Any],
        config: RunnableConfig,
        elapsed: float,
    ) -> dict[str, Any]:
        """Judge what the fixer left in the main checkout's working tree (before tests run).

        Reads the change set from git; handles *unfixable*, *no change* and the integrity
        check. Returns a state update; an attempt that is still unjudged afterwards (last
        history entry with ``accepted is None``) goes on to :func:`judge_and_commit`.
        """
        repo = _ctx(config).repo
        files = repo.changed_files()
        if sorted(set(attempt.files_changed)) != files:
            log.debug("[fix] fixer reported %s, git shows %s", attempt.files_changed, files)
        attempt = attempt.model_copy(update={"node_id": nid, "attempt": n, "files_changed": files})
        log.info(
            "[fix] %s: %s in %.1fs%s",
            nid,
            attempt.outcome,
            elapsed,
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
        report, failure = run_check(repo, nid, attempt.explanation)
        if report is None:
            repo.rollback()
            reason = f"{INTEGRITY_PREFIX} checker error: {failure}"
            return reject(state, previous, attempt, reason, {**update, "patch_report": None})
        if not report.ok:
            log.warning("[integrity] %s: %s", nid, ", ".join(report.rule_ids))
            repo.rollback()
            reason = f"{INTEGRITY_PREFIX}\n{report.summary()}"
            return reject(state, previous, attempt, reason, {**update, "patch_report": None})
        history = {**state.history, nid: [*previous, attempt]}
        return {**update, "history": history, "patch_report": report}

    def verify_one(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        assert state.current is not None
        return judge_and_commit(state, state.current, config)

    def judge_and_commit(state: PipelineState, nid: str, config: RunnableConfig) -> dict[str, Any]:
        """Verify the checked attempt for ``nid`` in the main checkout; commit or roll back.

        Re-runs all requested tests (flaky re-runs), asks the optional reviewer, runs the
        regression check for source/shared-test changes, then checkpoints an accepted attempt
        (marking side-effect fixes) or rolls a rejected one back. Used by the sequential
        ``verify_one`` and by the parallel ``merge_candidates``.
        """
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
        report = state.patch_report or PatchReport()
        if not reasons:
            rejected = review(state, nid, n, earlier, attempt, report, config)
            if rejected:
                reasons = [rejected]
        regression: dict[str, Any] = {}
        needs_full_suite = bool(
            report.source_files_changed or report.shared_test_files_changed
        )  # source, or shared test code (conftest/helpers/data) other tests may use
        if not reasons and needs_full_suite and not state.regression_unavailable:
            rejected, regression = regression_check(state, config)
            if rejected:
                reasons = [rejected]
        if reasons:
            repo.rollback()  # HEAD (and so ``state.results``) is the pre-attempt state
            update = {"patch_report": None, **regression}
            return reject(state, earlier, attempt, "; ".join(reasons), update)

        repo.checkpoint(f"ci-fix: fix {nid} (attempt {n})")
        files = attempt.files_changed
        log.info("[verify] %s fixed after %d attempt(s)", nid, n)
        outcomes = dict(state.outcomes)
        _set_outcome(
            outcomes,
            state,
            nid,
            OutcomeStatus.FIXED,
            "",
            n,
            files,
            explanation=attempt.explanation,
            test_changes=[c.describe() for c in report.expectation_changes],
            source_changed=bool(report.source_files_changed),
        )
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
            "patch_report": None,
            **regression,
        }

    def full_suite(
        config: RunnableConfig, keep: frozenset[str] | set[str] = frozenset()
    ) -> dict[str, TestStatus]:
        """Run the whole suite; files the run creates are excluded as artifacts.

        Paths in ``keep`` are never excluded (the stashed attempt's own new files).
        """
        repo = _ctx(config).repo
        untracked_before = repo.untracked_files()
        try:
            run = _runner(config).run_all(
                settings.regression_pytest_args, settings.regression_timeout_seconds
            )
        finally:
            repo.exclude(repo.untracked_files() - untracked_before - keep)
        return {tid: res.status for tid, res in run.results.items()}

    def restore_attempt(repo: GitRepo, attempt_untracked: set[str], changed: list[str]) -> str:
        """Pop the stashed attempt after the baseline run; return an error ("" = restored).

        Whatever the run changed is undone first: tracked files are reset to HEAD and files
        it created at the attempt's own new paths are deleted (they would block the pop).
        On failure the working tree is reset to HEAD (checkpoints are untouched) and the
        stash is dropped, so the attempt is lost and must be rejected.
        """
        try:
            repo.reset_hard()
            for rel in attempt_untracked:
                path = repo.path / rel
                if path.is_file() or path.is_symlink():
                    path.unlink()
            repo.unstash()
            now = repo.changed_files()
            if now != changed:
                return f"changed files {now} do not match the attempt's {changed}"
            return ""
        except (GitError, OSError) as exc:
            return f"{type(exc).__name__}: {exc}"

    def discard_attempt(repo: GitRepo) -> None:
        """Reset to HEAD and drop a leftover stash, after the attempt could not be restored."""
        try:
            repo.rollback()
            if repo.has_stash():
                repo.drop_stash()
        except GitError as exc:  # rejecting anyway; verify_one rolls back again
            log.error("[regression] Could not clean up after a failed restore: %s", exc)

    def regression_baseline(
        config: RunnableConfig,
    ) -> tuple[dict[str, TestStatus] | None, str, str]:
        """Full-suite statuses at HEAD, without the attempt's changes (stashed meanwhile).

        Returns ``(baseline or None, run error, restore error)``.
        """
        repo = _ctx(config).repo
        log.info("[regression] Source files changed: running the full suite without the fix")
        changed = repo.changed_files()
        # The attempt's new files (intent-to-add entries by now, so not "untracked").
        attempt_untracked = repo.added_files()
        stashed = repo.stash_all()
        baseline, run_error, restore_error = None, "", ""
        try:
            baseline = full_suite(config, keep=attempt_untracked)
        except TestRunError as exc:
            run_error = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            log.debug("[regression] baseline error: %s", exc)
        finally:
            if stashed:
                restore_error = restore_attempt(repo, attempt_untracked, changed)
                if restore_error:
                    log.error("[regression] Could not restore the attempt: %s", restore_error)
                    discard_attempt(repo)
        if baseline is not None:
            failing = sum(1 for st in baseline.values() if st in _FAILING)
            passed = sum(1 for st in baseline.values() if st == TestStatus.PASSED)
            log.info(
                "[regression] Baseline: %d passed, %d failing before any source change",
                passed,
                failing,
            )
        return baseline, run_error, restore_error

    def skip_regression(state: PipelineState, message: str) -> dict[str, Any]:
        """State update that turns regression checks off for the rest of the run."""
        log.warning("[regression] Regression check skipped for the rest of the run: %s", message)
        warning = f"{REGRESSION_SKIPPED_PREFIX} {message}"
        return {"regression_unavailable": True, "warnings": [*state.warnings, warning]}

    def regression_check(
        state: PipelineState, config: RunnableConfig
    ) -> tuple[str, dict[str, Any]]:
        """Full-suite check of an attempt that changed source files.

        Returns ``(rejection reason or "", state update)``. The update carries the baseline
        (computed here the first time), or turns the check off when a full run fails.
        """
        update: dict[str, Any] = {}
        baseline = state.regression_baseline
        if baseline is None:
            baseline, run_error, restore_error = regression_baseline(config)
            if baseline is not None:
                update["regression_baseline"] = baseline
            else:
                update = skip_regression(state, run_error)
            if restore_error:
                return f"{RESTORE_FAILED_PREFIX} {restore_error}", update
            if baseline is None:
                return "", update
        requested = _unique_ids(state)
        try:
            current = full_suite(config)
            regressed = find_regressions(baseline, current, requested)
            if regressed:
                # Re-run once so a flaky test doesn't sink a good fix.
                rerun = _runner(config).run(
                    regressed,
                    extra_args=settings.regression_pytest_args,
                    timeout=settings.regression_timeout_seconds,
                )
                still = []
                for tid in regressed:
                    res = rerun.results.get(tid)
                    status = res.status if res is not None else TestStatus.NOT_FOUND
                    if status in _FAILING or status == TestStatus.NOT_FOUND:
                        still.append(tid)
                        if status == TestStatus.NOT_FOUND:
                            current.pop(tid, None)
                    else:
                        log.info("[regression] %s passed on re-run; treating as flaky", tid)
                        current[tid] = status
                regressed = still
        except TestRunError as exc:
            message = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            log.debug("[regression] full-suite error: %s", exc)
            return "", {**update, **skip_regression(state, message)}
        if regressed:
            failing = [t for t in regressed if t in current]
            missing = [t for t in regressed if t not in current]
            log.warning("[regression] %d test(s) broke: %s", len(regressed), ", ".join(regressed))
            return regression_reason(failing, missing), update
        log.info("[regression] No regressions in the full suite")
        return "", {**update, "regression_baseline": current}

    def review(
        state: PipelineState,
        nid: str,
        n: int,
        earlier: list[FixAttempt],
        attempt: FixAttempt,
        report: PatchReport,
        config: RunnableConfig,
    ) -> str:
        """Ask the reviewer about test-file changes; return a rejection reason ("" = ok)."""
        if deps.reviewer is None or not report.test_files_changed:
            return ""
        repo = _ctx(config).repo
        diff = repo.diff("HEAD")  # the attempt only: HEAD is the last checkpoint
        request = make_request(state, nid, n, earlier, config)
        log.info(
            "[review] %s: reviewing test changes in %s", nid, ", ".join(report.test_files_changed)
        )
        verdict, error = None, ""
        for n_try in range(1, REVIEW_TRIES + 1):
            try:
                verdict = deps.reviewer.review(request, diff, attempt.explanation)
                break
            except FixerFatalError:
                repo.rollback()
                raise
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                log.warning(
                    "[review] reviewer error for %s (try %d/%d): %s",
                    nid,
                    n_try,
                    REVIEW_TRIES,
                    error,
                )
                log.debug("[review] reviewer traceback", exc_info=True)
        if verdict is None:  # when in doubt, reject
            return f"{REVIEWER_UNAVAILABLE_PREFIX} {error}"
        if verdict.approved:
            return ""
        log.warning("[review] %s: test change rejected: %s", nid, verdict.reason)
        return f"{REVIEWER_PREFIX} {verdict.reason}"

    # ---- parallel fixing ----------------------------------------------------------------

    def round_tests(state: PipelineState) -> list[str]:
        """The first pending test of each group, for up to ``max_parallel_workers`` groups."""
        return [g[0] for g in groups_of(state)][: settings.max_parallel_workers]

    def plan_round(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        assert state.prepared is not None
        n_round = state.parallel_round + 1
        tests = round_tests(state)
        pr_diff(state.prepared, _ctx(config))  # computed once here, not in the threads
        log.info(
            "[parallel] round %d: fixing %d test(s) in parallel: %s",
            n_round,
            len(tests),
            ", ".join(tests),
        )
        return {"parallel_round": n_round, "current": None, "candidates": []}

    def fan_out(state: PipelineState) -> list[Send]:
        return [
            Send("fix_in_worktree", state.model_copy(update={"current": nid}))
            for nid in round_tests(state)
        ]

    def worktree_dir(prepared: PreparedRepo, name: str) -> Path:
        """``run_dir/worktrees/<name>``, refusing anything outside the run dir."""
        root = (prepared.run_dir / "worktrees").resolve()
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError(f"worktree path {path} is outside {root}")
        return path

    def fix_in_worktree(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        """Run the fixer for ``state.current`` in its own worktree; return its patch.

        Runs in a thread next to the other branches of the round: it only touches its own
        worktree and runner (and appends artifact paths to the shared ``info/exclude``).
        """
        assert state.prepared is not None and state.current is not None
        nid = state.current
        n = state.attempts.get(nid, 0) + 1
        previous = state.history.get(nid, [])
        main = _ctx(config).repo
        index = round_tests(state).index(nid) + 1
        name = f"r{state.parallel_round}-{index}"
        path = worktree_dir(state.prepared, name)
        log.info("[fix] %s (attempt %d/%d, in parallel)", nid, n, settings.max_attempts)
        added = False
        try:
            worktree = main.add_worktree(path)
            added = True
            log.debug("[parallel] %s: worktree %s", nid, path)
            assert state.python is not None
            runner = deps.make_worktree_runner(
                path, state.python, state.prepared.reports_dir / "worktrees" / name
            )
            artifacts: set[str] = set()  # this branch's test-run artifacts, kept out of the patch
            run_test = make_run_test(nid, config, worktree, runner, artifacts.update)
            request = make_request(state, nid, n, previous, config, path, run_test)
            started = time.monotonic()
            attempt = deps.fixer.fix(request)
            elapsed = time.monotonic() - started
            patch = worktree.patch("HEAD", skip=artifacts)
            log.debug("[parallel] %s: patch of %d byte(s)", nid, len(patch))
            candidate = Candidate(node_id=nid, attempt=attempt, patch=patch, elapsed=elapsed)
            return {"candidates": [candidate]}
        except FixerFatalError as exc:
            log.error("[fix] stopping the run: %s", exc)
            raise
        except Exception as exc:  # a fixer (or worktree) error must not crash the run
            log.error("[fix] fixer raised for %s: %s", nid, exc, exc_info=True)
            return {"candidates": [Candidate(node_id=nid, error=str(exc) or type(exc).__name__)]}
        finally:
            if added:
                main.remove_worktree(path)
                log.debug("[parallel] %s: removed worktree %s", nid, path)

    def merge_candidates(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        """Apply and judge the round's candidates one by one, in ``pending`` order."""
        assert state.prepared is not None
        repo = _ctx(config).repo
        order = {nid: i for i, nid in enumerate(state.pending)}
        candidates = sorted(state.candidates, key=lambda c: order.get(c.node_id, len(order)))
        serial_queue: list[str] = []
        merged: dict[str, Any] = {}

        def advance(update: dict[str, Any]) -> None:
            nonlocal state
            merged.update(update)
            state = state.model_copy(update=update)

        for i, cand in enumerate(candidates, 1):
            nid = cand.node_id
            if nid not in state.pending:  # fixed as a side effect of an earlier candidate
                log.info("[parallel] %s already passes; candidate not needed", nid)
                continue
            log.info("[parallel] merging candidate for %s", nid)
            n = state.attempts.get(nid, 0) + 1
            previous = state.history.get(nid, [])
            update: dict[str, Any] = {"attempts": {**state.attempts, nid: n}}
            if cand.attempt is None:
                advance(reject_fixer_error(state, nid, n, previous, cand.error or "", update))
                continue
            if cand.attempt.outcome != "unfixable" and cand.patch:
                patch_file = (
                    state.prepared.reports_dir
                    / "worktrees"
                    / f"merge-r{state.parallel_round}-{i}.patch"
                )
                patch_file.parent.mkdir(parents=True, exist_ok=True)
                patch_file.write_bytes(cand.patch)
                try:
                    repo.apply_patch(patch_file)
                except GitError as exc:
                    log.debug("[parallel] apply failed for %s: %s", nid, exc)
                    log.info(
                        "[parallel] patch for %s conflicts with an accepted fix; "
                        "retrying sequentially",
                        nid,
                    )
                    # The main checkout was clean before the apply (every earlier candidate
                    # ended in a checkpoint commit or a rollback) and ``git apply`` is atomic,
                    # so this rollback is a no-op safety net; it never touches accepted fixes.
                    repo.rollback()
                    serial_queue.append(nid)
                    continue
            advance(
                after_fixer(state, nid, n, previous, cand.attempt, update, config, cand.elapsed)
            )
            last = state.history.get(nid, [])
            if last and last[-1].accepted is None and last[-1].outcome != "unfixable":
                advance(judge_and_commit(state, nid, config))
        return {**merged, "candidates": [], "current": None, "serial_queue": serial_queue}

    def finalize(state: PipelineState, config: RunnableConfig) -> dict[str, Any]:
        diff = ""
        if state.prepared is not None:
            # The worktree is clean: the diff is exactly the accepted checkpoint commits.
            diff = _ctx(config).repo.diff(state.prepared.pr_head_sha)
        summary = build_summary(state)
        preexisting: list[str] = []
        if state.regression_baseline is not None:
            preexisting = preexisting_failures(state.regression_baseline, _unique_ids(state))
            if preexisting:
                log.info(
                    "[finalize] %d pre-existing failure(s) outside the requested tests",
                    len(preexisting),
                )
        counts: dict[str, int] = {}
        for outcome in state.outcomes.values():
            counts[outcome.status.value] = counts.get(outcome.status.value, 0) + 1
        log.info(
            "[finalize] %s; diff %d line(s)",
            ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "no outcomes",
            len(diff.splitlines()),
        )
        return {"diff": diff, "summary": summary, "preexisting_failures": preexisting}

    def after_resolve(state: PipelineState) -> str:
        return "run_initial" if state.name_to_id else "finalize"

    def after_select(state: PipelineState) -> str:
        if state.current is not None:
            return "fix_one"
        return "plan_round" if state.pending else "finalize"

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
    graph.add_node("plan_round", plan_round)
    graph.add_node("fix_in_worktree", fix_in_worktree)
    graph.add_node("merge_candidates", merge_candidates)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "setup_repo")
    graph.add_edge("setup_repo", "setup_env")
    graph.add_edge("setup_env", "resolve_tests")
    graph.add_conditional_edges("resolve_tests", after_resolve, ["run_initial", "finalize"])
    graph.add_edge("run_initial", "select_next")
    graph.add_conditional_edges("select_next", after_select, ["fix_one", "plan_round", "finalize"])
    graph.add_conditional_edges("fix_one", after_fix, ["verify_one", "select_next"])
    graph.add_edge("verify_one", "select_next")
    graph.add_conditional_edges("plan_round", fan_out, ["fix_in_worktree"])
    graph.add_edge("fix_in_worktree", "merge_candidates")
    graph.add_edge("merge_candidates", "select_next")
    graph.add_edge("finalize", END)
    return graph.compile()
