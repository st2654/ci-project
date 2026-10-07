"""Helpers for the slice-7 parallel-fixing tests.

* A sample remote with a second module (``src/calc/strings.py`` + ``tests/test_strings.py``)
  so two seeded bugs live in different source and test files (independent failures).
* A ``FakeRunner`` variant whose failure details are scripted pytest tracebacks, so the
  triage step sees which files each failure touches.
* A thread-safe fixer that records when, where (``repo_path``) and on which thread it ran.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from conftest import git
from fixtures import copy_sample_repo
from pipeline_helpers import (
    OPS_PY,
    SAMPLE_PR,
    SAMPLE_RULES,
    SUBTRACT,
    FakeRunner,
    FakeRunnerFactory,
    Rule,
    SampleRemote,
    ScriptedFixer,
)

from ci_fix.config import Settings
from ci_fix.models import FixAttempt, FixerFatalError, FixRequest
from ci_fix.tools.pytest_runner import TestResult, TestStatus

STRINGS_PY = "src/calc/strings.py"
TEST_STRINGS_PY = "tests/test_strings.py"
SHOUT = f"{TEST_STRINGS_PY}::test_shout"

STRINGS_SOURCE = '''"""String helpers. Contains a deliberately seeded bug."""


def shout(text: str) -> str:
    """Return the text upper-cased, followed by "!"."""
    return text.lower() + "!"  # SEEDED BUG: should be text.upper()
'''
SHOUT_TEST = 'def test_shout():\n    assert shout("hi") == "HI!"\n'
TEST_STRINGS_SOURCE = f"from calc.strings import shout\n\n\n{SHOUT_TEST}"
SHOUT_BUG = 'return text.lower() + "!"  # SEEDED BUG: should be text.upper()'
SHOUT_FIX = 'return text.upper() + "!"'

FIX_SHOUT = ("edit", STRINGS_PY, SHOUT_BUG, SHOUT_FIX, "shout lower-cased instead")
DELETE_SHOUT_TEST = ("edit", TEST_STRINGS_PY, SHOUT_TEST, "", "removed the test")

RULES: dict[str, Rule] = {
    **SAMPLE_RULES,
    SHOUT: lambda read: SHOUT_FIX in read(STRINGS_PY),
}

# Tracebacks as pytest prints them (``--tb=short``): independent failures.
TB_SUBTRACT = (
    "tests/test_ops.py:9: in test_subtract\n"
    "    assert subtract(5, 3) == 2\n"
    "E   assert 8 == 2\n"
    "src/calc/ops.py:11: AssertionError\n"
)
TB_SHOUT = (
    "tests/test_strings.py:5: in test_shout\n"
    '    assert shout("hi") == "HI!"\n'
    "src/calc/strings.py:6: AssertionError\n"
)
# SHOUT's traceback also goes through ops.py: dependent on SUBTRACT.
TB_SHOUT_VIA_OPS = TB_SHOUT + "src/calc/ops.py:5: in add\n"
TRACEBACKS = {SUBTRACT: TB_SUBTRACT, SHOUT: TB_SHOUT}


def make_two_module_remote(tmp_path: Path, pr_numbers=(SAMPLE_PR,)) -> SampleRemote:
    """Like ``make_sample_remote`` plus ``strings.py``/``test_strings.py`` (one seeded bug)."""
    work = copy_sample_repo(tmp_path / "two-module-origin")
    (work / STRINGS_PY).write_text(STRINGS_SOURCE, encoding="utf-8")
    (work / TEST_STRINGS_PY).write_text(TEST_STRINGS_SOURCE, encoding="utf-8")
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

    bare = tmp_path / "two-module-remote.git"
    git("clone", "-q", "--bare", str(work), str(bare))
    for number in pr_numbers:
        git("update-ref", f"refs/pull/{number}/head", pr_sha, cwd=bare)
    return SampleRemote(bare=bare, pr_number=pr_numbers[0], pr_sha=pr_sha)


# ---- runner with traceback details ----------------------------------------------------------


@dataclass
class TracebackRunner(FakeRunner):
    """``FakeRunner`` whose failing results carry a scripted traceback as ``details``.

    Like ``FakeRunner`` it reads files from its own ``repo_path`` (main checkout or worktree).
    """

    tracebacks: dict[str, str] = field(default_factory=dict)

    def _result(self, node_id: str) -> TestResult:
        result = super()._result(node_id)
        if result.status in (TestStatus.FAILED, TestStatus.ERROR) and node_id in self.tracebacks:
            details = f"{result.details}\n{self.tracebacks[node_id]}"
            result = result.model_copy(update={"details": details})
        return result


@dataclass
class TracebackRunnerFactory(FakeRunnerFactory):
    tracebacks: dict[str, str] = field(default_factory=lambda: dict(TRACEBACKS))

    def __call__(self, repo_path: Path, python: Path, reports_dir: Path, settings: Settings):
        runner = TracebackRunner(
            Path(repo_path), self.rules, self.artifacts, tracebacks=self.tracebacks
        )
        with _FACTORY_LOCK:
            self.runners.append(runner)
        return runner


_FACTORY_LOCK = threading.Lock()


# ---- concurrent fixer -----------------------------------------------------------------------


@dataclass(frozen=True)
class Call:
    node_id: str
    attempt: int
    repo_path: Path
    thread: int
    start: float
    end: float


def overlapped(a: Call, b: Call) -> bool:
    return a.start < b.end and b.start < a.end


def max_concurrency(calls: list[Call]) -> int:
    events = sorted([(c.start, 1) for c in calls] + [(c.end, -1) for c in calls])
    current = best = 0
    for _, delta in events:
        current += delta
        best = max(best, current)
    return best


class ConcurrentFixer:
    """Thread-safe scripted fixer. Edits are applied inside ``request.repo_path``.

    Script actions are ``ScriptedFixer``'s plus ``("fatal", message)`` (raises
    ``FixerFatalError``). ``delay`` seconds are slept before acting, to make overlap visible.
    """

    def __init__(self, script: dict[str, list[tuple]], delay: float = 0.0) -> None:
        self.script = script
        self.delay = delay
        self.calls: list[Call] = []
        self.requests: list[FixRequest] = []
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def calls_for(self, node_id: str) -> list[FixRequest]:
        with self._lock:
            return [r for r in self.requests if r.node_id == node_id]

    def fix(self, request: FixRequest) -> FixAttempt:
        start = time.monotonic()
        nid = request.node_id
        with self._lock:
            index = self._counts.get(nid, 0)
            self._counts[nid] = index + 1
            self.requests.append(request)
        try:
            if self.delay:
                time.sleep(self.delay)
            actions = self.script.get(nid, [])
            action = actions[index] if index < len(actions) else ("noop",)
            if action[0] == "fatal":
                raise FixerFatalError(action[1])
            return ScriptedFixer({nid: [action]}).fix(request)
        finally:
            call = Call(
                nid,
                request.attempt,
                Path(request.repo_path),
                threading.get_ident(),
                start,
                time.monotonic(),
            )
            with self._lock:
                self.calls.append(call)


__all__ = [
    "DELETE_SHOUT_TEST",
    "FIX_SHOUT",
    "OPS_PY",
    "RULES",
    "SHOUT",
    "SHOUT_BUG",
    "SHOUT_FIX",
    "STRINGS_PY",
    "TB_SHOUT",
    "TB_SHOUT_VIA_OPS",
    "TB_SUBTRACT",
    "TEST_STRINGS_PY",
    "TRACEBACKS",
    "Call",
    "ConcurrentFixer",
    "TracebackRunner",
    "TracebackRunnerFactory",
    "make_two_module_remote",
    "max_concurrency",
    "overlapped",
]
