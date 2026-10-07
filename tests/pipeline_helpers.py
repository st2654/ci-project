"""Test harness for the pipeline: a fake remote built from the sample repo, ``PipelineDeps``
that never touch the network or install anything, a scripted fixer and a fake test runner.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from conftest import git
from fixtures import copy_sample_repo

from ci_fix.config import Settings
from ci_fix.graph import PipelineDeps
from ci_fix.models import FixAttempt, FixRequest
from ci_fix.tools.github import PullRequestInfo
from ci_fix.tools.pytest_runner import TestResult, TestRunResult, TestStatus
from ci_fix.tools.test_env import TestEnv

SAMPLE_PR = 11
OTHER_PR = 12  # same commit as SAMPLE_PR; lets two runs share one remote
REPO_URL = "https://github.com/octo/sample"
OPS = "tests/test_ops.py"
ERRS = "tests/test_errors.py"
OPS_PY = "src/calc/ops.py"
TEST_OPS_PY = "tests/test_ops.py"

ADD = f"{OPS}::test_add"
SUBTRACT = f"{OPS}::test_subtract"
DIV_OK = f"{OPS}::TestDivide::test_divide_ok"
DIV_ZERO = f"{OPS}::TestDivide::test_divide_by_zero"
MEAN = f"{OPS}::test_mean"
SKIPPED = f"{OPS}::test_skipped"
ERRS_ADD = f"{ERRS}::test_add"
FIXTURE_ERROR = f"{ERRS}::test_fixture_error"

# One-line fixes for the seeded bugs (old text is unique in ops.py).
ADD_BODY = '"""Return a + b."""\n    return a + b\n'
SUBTRACT_BUG = "return a + b  # SEEDED BUG: should be a - b"
SUBTRACT_FIX = "return a - b"
DIVIDE_BUG = "return a / b  # SEEDED BUG: missing the b == 0 check that raises ValueError"
DIVIDE_FIX = 'if b == 0:\n        raise ValueError("division by zero")\n    return a / b'
MEAN_BUG = "return sum(xs) / (len(xs) + 1)  # SEEDED BUG: off-by-one, should be len(xs)"
MEAN_FIX = "return sum(xs) / len(xs)"
SUBTRACT_TEST = "def test_subtract():\n    assert subtract(5, 3) == 2\n"

# Scripted fixer actions for the seeded bugs.
FIX_SUBTRACT = ("edit", OPS_PY, SUBTRACT_BUG, SUBTRACT_FIX, "subtract added instead")
FIX_DIVIDE = ("edit", OPS_PY, DIVIDE_BUG, DIVIDE_FIX, "divide lacked the zero check")
FIX_MEAN = ("edit", OPS_PY, MEAN_BUG, MEAN_FIX, "mean divided by len + 1")
WRONG_SUBTRACT = ("edit", OPS_PY, SUBTRACT_BUG, "return a * b", "try multiplication")
DELETE_SUBTRACT_TEST = ("edit", TEST_OPS_PY, SUBTRACT_TEST, "", "removed the test")


# ---- sample remote --------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleRemote:
    """A bare repo containing the sample repo, with ``refs/pull/<N>/head`` set."""

    bare: Path
    pr_number: int
    pr_sha: str

    @property
    def url(self) -> str:
        return str(self.bare)


def make_sample_remote(tmp_path: Path, pr_numbers: Sequence[int] = (SAMPLE_PR, OTHER_PR)):
    """Build the remote. Tests must never write to it (they clone it), so it can be shared."""
    work = copy_sample_repo(tmp_path / "sample-origin")
    git("init", "-q", "-b", "main", cwd=work)
    git("config", "user.email", "tester@example.com", cwd=work)
    git("config", "user.name", "Tester", cwd=work)
    git("config", "commit.gpgsign", "false", cwd=work)
    git("add", "-A", cwd=work)
    git("commit", "-q", "-m", "initial", cwd=work)

    git("checkout", "-q", "-b", "feature", cwd=work)
    (work / "PR_CHANGE.txt").write_text("change from the PR\n", encoding="utf-8")
    git("add", "PR_CHANGE.txt", cwd=work)
    git("commit", "-q", "-m", "pr change", cwd=work)
    pr_sha = git("rev-parse", "HEAD", cwd=work)
    git("checkout", "-q", "main", cwd=work)

    bare = tmp_path / "sample-remote.git"
    git("clone", "-q", "--bare", str(work), str(bare))
    for number in pr_numbers:
        git("update-ref", f"refs/pull/{number}/head", pr_sha, cwd=bare)
    return SampleRemote(bare=bare, pr_number=pr_numbers[0], pr_sha=pr_sha)


def sample_pr_info(remote: SampleRemote, number: int | None = None) -> PullRequestInfo:
    number = number or remote.pr_number
    return PullRequestInfo(
        number=number,
        title="Feature",
        state="open",
        base_ref="main",
        head_ref="feature",
        head_sha=remote.pr_sha,
        head_repo_full_name="octo/sample",
        is_fork=False,
        html_url=f"{REPO_URL}/pull/{number}",
    )


def fake_env_factory(repo_path: Path, venv_dir: Path, settings: Settings) -> TestEnv:
    """Stand-in for ``create_test_env``: reuse the current interpreter, install nothing."""
    return TestEnv(venv_dir=Path(venv_dir), python=Path(sys.executable), install_method="none")


def make_deps(
    tmp_path: Path,
    fixer: Any,
    *,
    remote: SampleRemote,
    runner_factory: Any = None,
    env_factory: Any = fake_env_factory,
    reviewer: Any = None,
    **settings_overrides: Any,
) -> PipelineDeps:
    """Deps for the sample remote. ``runner_factory=None`` runs the real pytest."""
    settings = Settings(workspace_dir=tmp_path / "ws", **settings_overrides)
    github = MagicMock()
    github.get_pull_request.side_effect = lambda ref, number: sample_pr_info(remote, number)
    extra = {"runner_factory": runner_factory} if runner_factory is not None else {}
    return PipelineDeps(
        settings=settings,
        github=github,
        fixer=fixer,
        env_factory=env_factory,
        clone_url=remote.url,
        reviewer=reviewer,
        **extra,
    )


def run_dirs_for(tmp_path: Path, pr_number: int = SAMPLE_PR) -> list[Path]:
    """Run dirs of ``pr_number`` that still exist (names end in a unique per-run suffix)."""
    return sorted((tmp_path / "ws").glob(f"octo__sample__pr-{pr_number}__*"))


def run_dir_for(tmp_path: Path, pr_number: int = SAMPLE_PR) -> Path:
    """The single existing run dir of ``pr_number``."""
    (run_dir,) = run_dirs_for(tmp_path, pr_number)
    return run_dir


# ---- fake test runner -----------------------------------------------------------------------

# A rule decides one test's status from the CURRENT file contents of the checkout.
# ``read(relpath)`` returns the file's text ("" if missing). True/False mean passed/failed.
Read = Callable[[str], str]
Rule = Callable[[Read], "bool | TestStatus"]


def _subtract_rule(read: Read) -> bool | TestStatus:
    tests = read(TEST_OPS_PY)
    if "def test_subtract" not in tests:
        return TestStatus.NOT_FOUND
    if "@pytest.mark.skip\ndef test_subtract" in tests:
        return TestStatus.SKIPPED
    return SUBTRACT_FIX in read(OPS_PY)


SAMPLE_RULES: dict[str, Rule] = {
    ADD: lambda read: ADD_BODY in read(OPS_PY),
    SUBTRACT: _subtract_rule,
    DIV_OK: lambda read: True,
    DIV_ZERO: lambda read: 'raise ValueError("division by zero")' in read(OPS_PY),
    MEAN: lambda read: MEAN_FIX in read(OPS_PY),
    SKIPPED: lambda read: TestStatus.SKIPPED,
    ERRS_ADD: lambda read: ADD_BODY in read(OPS_PY),
    FIXTURE_ERROR: lambda read: TestStatus.ERROR,
}


def flaky(rule: Rule, overrides: dict[int, bool | TestStatus]) -> Rule:
    """``rule``, except on the given (1-based) evaluations, which return the override."""
    calls = 0

    def evaluate(read: Read) -> bool | TestStatus:
        nonlocal calls
        calls += 1
        return overrides[calls] if calls in overrides else rule(read)

    return evaluate


def marker_rule(rule: Rule, markers: dict[str, bool | TestStatus]) -> Rule:
    """``rule``, unless ops.py contains one of ``markers`` (then that marker's status)."""

    def evaluate(read: Read) -> bool | TestStatus:
        ops = read(OPS_PY)
        for marker, status in markers.items():
            if marker in ops:
                return status
        return rule(read)

    return evaluate


@dataclass
class FakeRunner:
    """In-memory stand-in for ``PytestRunner``: statuses come from ``rules``, not pytest."""

    repo_path: Path
    rules: dict[str, Rule]
    artifacts: bool = False  # each run leaves .coverage and out/run-<n>.log behind
    collection_errors: set[str] = field(default_factory=set)
    runs: list[list[str]] = field(default_factory=list)
    full_runs: int = 0  # run_all calls (not recorded in ``runs``)
    run_options: list[tuple[list[str], float | None]] = field(default_factory=list)  # per run

    def _read(self, relpath: str) -> str:
        path = self.repo_path / relpath
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    def collect(self) -> list[str]:
        return list(self.rules)

    def _result(self, node_id: str) -> TestResult:
        rule = self.rules.get(node_id)
        verdict = rule(self._read) if rule is not None else TestStatus.NOT_FOUND
        if isinstance(verdict, bool):
            verdict = TestStatus.PASSED if verdict else TestStatus.FAILED
        message = "" if verdict == TestStatus.PASSED else f"{verdict.value}: {node_id}"
        return TestResult(node_id=node_id, status=verdict, message=message, details=message)

    def run(
        self, node_ids: Sequence[str], extra_args: Sequence[str] = (), timeout: float | None = None
    ) -> TestRunResult:
        ids = list(dict.fromkeys(node_ids))
        self.runs.append(ids)
        self.run_options.append((list(extra_args), timeout))
        if self.artifacts:
            (self.repo_path / ".coverage").write_text(f"run {len(self.runs)}\n")
            (self.repo_path / "out").mkdir(exist_ok=True)
            (self.repo_path / "out" / f"run-{len(self.runs)}.log").write_text("log\n")
        results = {nid: self._result(nid) for nid in ids}
        failed = any(r.status != TestStatus.PASSED for r in results.values())
        return TestRunResult(results=results, exit_code=int(failed), duration=0.0, output_tail="")

    def run_all(
        self, extra_args: Sequence[str] = (), timeout: float | None = None
    ) -> TestRunResult:
        """The whole "suite": every rule's test that is currently collected."""
        self.full_runs += 1
        results = {nid: self._result(nid) for nid in self.rules}
        results = {k: v for k, v in results.items() if v.status != TestStatus.NOT_FOUND}
        failed = any(r.status != TestStatus.PASSED for r in results.values())
        return TestRunResult(results=results, exit_code=int(failed), duration=0.0, output_tail="")


@dataclass
class FakeRunnerFactory:
    """``PipelineDeps.runner_factory`` that builds a ``FakeRunner`` per run."""

    rules: dict[str, Rule] = field(default_factory=lambda: dict(SAMPLE_RULES))
    artifacts: bool = False
    runners: list[FakeRunner] = field(default_factory=list)

    def __call__(self, repo_path: Path, python: Path, reports_dir: Path, settings: Settings):
        runner = FakeRunner(Path(repo_path), self.rules, self.artifacts)
        self.runners.append(runner)
        return runner


# ---- scripted fixer -------------------------------------------------------------------------

Action = tuple[Any, ...]


def _edit(repo: Path, relpath: str, old: str, new: str) -> None:
    path = repo / relpath
    text = path.read_text(encoding="utf-8")
    assert old in text, f"scripted edit: {old!r} not found in {relpath}"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


@dataclass
class ScriptedFixer:
    """A fixer that follows a per-test script of actions, one per attempt.

    Actions:
    ``("edit", relpath, old, new, explanation)``;
    ``("edits", [(relpath, old, new), ...], explanation)``;
    ``("noop",)``; ``("unfixable", reason)``; ``("raise", message)``;
    ``("edit_then_unfixable", relpath, old, new, reason)``;
    ``("edit_then_raise", relpath, old, new, message)``.
    Running out of script means ``noop``.
    """

    script: dict[str, list[Action]] = field(default_factory=dict)
    requests: list[FixRequest] = field(default_factory=list)

    def calls_for(self, node_id: str) -> list[FixRequest]:
        return [r for r in self.requests if r.node_id == node_id]

    def fix(self, request: FixRequest) -> FixAttempt:
        self.requests.append(request)
        actions = self.script.get(request.node_id, [])
        index = len(self.calls_for(request.node_id)) - 1
        action = actions[index] if index < len(actions) else ("noop",)
        kind, repo = action[0], Path(request.repo_path)

        def attempt(outcome: str, explanation: str = "", files: Sequence[str] = ()) -> FixAttempt:
            return FixAttempt(
                node_id=request.node_id,
                attempt=request.attempt,
                outcome=outcome,
                explanation=explanation,
                files_changed=list(files),
            )

        if kind == "edit":
            _, relpath, old, new, explanation = action
            _edit(repo, relpath, old, new)
            return attempt("changed", explanation, [relpath])
        if kind == "edits":
            _, edits, explanation = action
            for relpath, old, new in edits:
                _edit(repo, relpath, old, new)
            return attempt("changed", explanation, sorted({e[0] for e in edits}))
        if kind == "edit_then_unfixable":
            _, relpath, old, new, reason = action
            _edit(repo, relpath, old, new)
            return attempt("unfixable", reason, [relpath])
        if kind == "edit_then_raise":
            _, relpath, old, new, message = action
            _edit(repo, relpath, old, new)
            raise RuntimeError(message)
        if kind == "unfixable":
            return attempt("unfixable", action[1])
        if kind == "raise":
            raise RuntimeError(action[1])
        return attempt("no_change")


@dataclass
class StatelessFixer:
    """Applies a fixed action per node id on every call; safe to share between threads."""

    actions: dict[str, Action]

    def fix(self, request: FixRequest) -> FixAttempt:
        return ScriptedFixer({request.node_id: [self.actions[request.node_id]]}).fix(request)
