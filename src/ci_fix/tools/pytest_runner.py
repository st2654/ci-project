"""Run pytest node ids in the target repo and parse the JUnit XML results."""

from __future__ import annotations

import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from ci_fix.logging_setup import get_logger
from ci_fix.tools.target_env import build_target_env, run_target

log = get_logger(__name__)

OUTPUT_TAIL_CHARS = 4000
_PARAM_RE = re.compile(r"\[.*\]$")

PLUGIN_MODULE = "_ci_fix_pytest_plugin"
NODEID_PROPERTY = "ci_fix_nodeid"
# Loaded into the target's pytest with ``-p``: records each item's real node id in the JUnit
# report, so ids are exact even for tests inherited from a base class in another file.
_PLUGIN_SOURCE = f"""\
\"\"\"Written by ci-fix: record each test's node id as a JUnit property.\"\"\"


def pytest_collection_modifyitems(items):
    for item in items:
        item.user_properties.append(({NODEID_PROPERTY!r}, item.nodeid))
"""


class TestRunError(Exception):
    """Raised when pytest cannot be run (collection failure, timeout, missing interpreter)."""

    __test__ = False


class TestStatus(StrEnum):
    __test__ = False

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    NOT_FOUND = "not_found"


# Worse statuses win when one test has several JUnit entries (e.g. failure + teardown error).
_SEVERITY = {
    TestStatus.PASSED: 0,
    TestStatus.SKIPPED: 1,
    TestStatus.FAILED: 2,
    TestStatus.ERROR: 3,
}


class TestResult(BaseModel):
    __test__ = False

    node_id: str
    status: TestStatus
    message: str = ""
    details: str = ""
    duration: float = 0.0


class TestRunResult(BaseModel):
    __test__ = False

    results: dict[str, TestResult]
    exit_code: int
    duration: float
    output_tail: str

    @property
    def failing(self) -> list[str]:
        bad = (TestStatus.FAILED, TestStatus.ERROR)
        return [nid for nid, r in self.results.items() if r.status in bad]

    @property
    def passed(self) -> list[str]:
        return [nid for nid, r in self.results.items() if r.status is TestStatus.PASSED]


class ResolvedTests(BaseModel):
    __test__ = False

    node_ids: list[str]
    ambiguous: dict[str, list[str]]
    not_found: list[str]


def _strip_param(node_id: str) -> str:
    return _PARAM_RE.sub("", node_id)


def _run(
    argv: Sequence[str], cwd: Path, env: Mapping[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    log.debug("$ %s (cwd=%s)", " ".join(argv), cwd)
    started = time.monotonic()
    try:
        proc = run_target(argv, cwd=cwd, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise TestRunError(f"pytest timed out after {timeout}s: {' '.join(argv)}") from None
    except OSError as exc:
        raise TestRunError(f"failed to run {' '.join(argv)}: {exc}") from None
    log.debug("pytest exited %d in %.2fs", proc.returncode, time.monotonic() - started)
    return proc


def _testcase_node_id(case: ET.Element) -> str:
    file = case.get("file", "")
    name = case.get("name", "")
    classname = case.get("classname", "")
    module = file.removesuffix(".py").replace("/", ".")
    if module and (classname == module or classname.startswith(module + ".")):
        class_path = classname[len(module) :].lstrip(".")
    elif not file:
        class_path = ""
    else:
        class_path = classname
    parts = [file] + [p for p in class_path.split(".") if p] + [name]
    return "::".join(parts)


def _first_error_line(details: str) -> str:
    """The first ``E   ...`` line of a pytest traceback (e.g. the ImportError)."""
    for line in details.splitlines():
        if line.startswith("E "):
            return line[1:].strip()
    return ""


def _file_of(node_id: str) -> str:
    return node_id.split("::", 1)[0]


def _recorded_node_id(case: ET.Element) -> str:
    """The node id our plugin stored in the testcase's properties ("" if absent)."""
    for prop in case.iter("property"):
        if prop.get("name") == NODEID_PROPERTY:
            return prop.get("value", "")
    return ""


def _merge(a: TestResult, b: TestResult) -> TestResult:
    """Combine two entries for one node id: keep the worse status, concatenate details."""
    worse, other = (b, a) if _SEVERITY[b.status] > _SEVERITY[a.status] else (a, b)
    details = "\n\n".join(d for d in (worse.details, other.details) if d)
    return worse.model_copy(update={"details": details, "duration": a.duration + b.duration})


def _parse_junit(path: Path) -> list[TestResult]:
    root = ET.parse(path).getroot()
    results: dict[str, TestResult] = {}
    for case in root.iter("testcase"):
        status, message, details = TestStatus.PASSED, "", ""
        for tag, st in (
            ("error", TestStatus.ERROR),
            ("failure", TestStatus.FAILED),
            ("skipped", TestStatus.SKIPPED),
        ):
            el = case.find(tag)
            if el is not None:
                status, message, details = st, el.get("message", ""), (el.text or "")
                break
        file_attr = case.get("file") or ""
        is_collection_error = status is TestStatus.ERROR and not case.get("classname")
        # File-level collection errors have no test item, hence no recorded node id.
        node_id = _recorded_node_id(case)
        if not node_id:
            node_id = file_attr if is_collection_error and file_attr else _testcase_node_id(case)
        if is_collection_error:
            message = f"{message}: {_first_error_line(details)}".rstrip(": ")
        result = TestResult(
            node_id=node_id,
            status=status,
            message=message,
            details=details,
            duration=float(case.get("time", "0") or 0),
        )
        results[node_id] = _merge(results[node_id], result) if node_id in results else result
    return list(results.values())


def _aggregate(requested: str, cases: list[TestResult]) -> TestResult:
    statuses = {c.status for c in cases}
    if TestStatus.ERROR in statuses:
        status = TestStatus.ERROR
    elif TestStatus.FAILED in statuses:
        status = TestStatus.FAILED
    elif statuses == {TestStatus.SKIPPED}:
        status = TestStatus.SKIPPED
    else:
        status = TestStatus.PASSED
    message, details = "", ""
    first_bad = next((c for c in cases if c.status is status), None)
    if status in (TestStatus.ERROR, TestStatus.FAILED) and first_bad is not None:
        param = first_bad.node_id[len(requested) :]
        message = f"{param} {first_bad.message}".strip()
        details = f"{param}\n{first_bad.details}" if first_bad.details else param
    return TestResult(
        node_id=requested,
        status=status,
        message=message,
        details=details,
        duration=sum(c.duration for c in cases),
    )


def _not_found(node_id: str, message: str = "not collected by pytest") -> TestResult:
    return TestResult(node_id=node_id, status=TestStatus.NOT_FOUND, message=message)


def _collection_error_path(line: str) -> str:
    """The path in a ``ERROR <path> [- <reason>]`` line (paths may contain spaces)."""
    rest = line[len("ERROR ") :]
    if " - " in rest:
        rest = rest.split(" - ", 1)[0]
    return rest.strip()


class PytestRunner:
    """Runs pytest in a repo with a given interpreter and reports per-test results.

    The target repo's code is untrusted, so pytest runs with an allowlisted environment
    (``build_target_env``) unless ``env`` is given. ``extra_args`` are passed to both
    collection and runs. Options that stop pytest early (``-x``, ``--maxfail``) can leave
    requested tests unrun; those are reported as NOT_FOUND with message
    ``"not run (pytest stopped early)"``.
    """

    def __init__(
        self,
        repo_path: Path,
        python: Path,
        reports_dir: Path,
        extra_args: Sequence[str] = (),
        timeout: float = 900,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.python = Path(python)
        self.reports_dir = Path(reports_dir)
        self.extra_args = list(extra_args)
        self.timeout = timeout
        self._runs = 0
        self.collection_errors: set[str] = set()
        self._collected: set[str] = set()
        self.plugin_dir = self.reports_dir / "_plugin"
        if env is None:
            self.env = build_target_env(self.python.parent.parent, [self.plugin_dir])
        else:
            self.env = dict(env)
            pythonpath = [p for p in self.env.get("PYTHONPATH", "").split(os.pathsep) if p]
            self.env["PYTHONPATH"] = os.pathsep.join([*pythonpath, str(self.plugin_dir)])

    def _ensure_plugin(self) -> None:
        self.plugin_dir.mkdir(parents=True, exist_ok=True)
        (self.plugin_dir / f"{PLUGIN_MODULE}.py").write_text(_PLUGIN_SOURCE, encoding="utf-8")

    def _pytest(self, *args: str, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        self._ensure_plugin()
        # --rootdir pins node ids to the repo root even when the repo has no pytest config;
        # otherwise ids would depend on which paths are passed (collect vs run).
        argv = [
            str(self.python),
            "-m",
            "pytest",
            "-p",
            PLUGIN_MODULE,
            f"--rootdir={self.repo_path}",
            *args,
        ]
        return _run(argv, self.repo_path, self.env, self.timeout if timeout is None else timeout)

    def _junit_args(self, junit: Path) -> list[str]:
        """Options shared by every reporting run (``run`` and ``run_all``)."""
        return [
            "--continue-on-collection-errors",
            f"--junitxml={junit}",
            "-o",
            "junit_family=xunit1",
            "-p",
            "no:cacheprovider",
            "-q",
            "-rN",
        ]

    def run_all(
        self, extra_args: Sequence[str] = (), timeout: float | None = None
    ) -> TestRunResult:
        """Run the whole test suite (what pytest's own config collects).

        Results are keyed by the real node ids; a file that fails to import is one ERROR
        entry keyed by its path. ``extra_args`` come after the runner's own ``extra_args``;
        ``timeout`` overrides the runner's timeout for this run. Raises ``TestRunError`` on a
        timeout or when pytest produces no usable report.
        """
        self._runs += 1
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        junit = self.reports_dir / f"full-{self._runs}.xml"
        junit.unlink(missing_ok=True)
        log.info("[regression] Running full test suite")
        started = time.monotonic()
        proc = self._pytest(
            *self._junit_args(junit), *self.extra_args, *extra_args, timeout=timeout
        )
        duration = time.monotonic() - started
        output_tail = (proc.stdout or "")[-OUTPUT_TAIL_CHARS:]
        cases: list[TestResult] | None = None
        if junit.is_file() and junit.stat().st_size > 0:
            try:
                cases = _parse_junit(junit)
            except ET.ParseError as exc:
                log.debug("Could not parse %s: %s", junit, exc)
        # Exit 5 = nothing collected (an empty suite is a valid result).
        if cases is None or (not cases and proc.returncode not in (0, 5)):
            raise TestRunError(
                f"pytest did not produce results for the full suite (exit {proc.returncode})"
                f"\n{output_tail}".rstrip()
            )
        results = {c.node_id: c for c in cases}
        counts = {st: sum(1 for r in results.values() if r.status is st) for st in TestStatus}
        log.info(
            "[regression] %d passed, %d failed, %d error, %d skipped in %.1fs",
            counts[TestStatus.PASSED],
            counts[TestStatus.FAILED],
            counts[TestStatus.ERROR],
            counts[TestStatus.SKIPPED],
            duration,
        )
        return TestRunResult(
            results=results, exit_code=proc.returncode, duration=duration, output_tail=output_tail
        )

    def collect(self, timeout: float | None = None) -> list[str]:
        """Return all node ids pytest collects in the repo.

        Test files that fail to import are recorded in ``collection_errors`` (their tests
        cannot be listed) instead of aborting collection.
        """
        proc = self._pytest(
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "--continue-on-collection-errors",
            *self.extra_args,
            timeout=timeout,
        )
        stdout = proc.stdout or ""
        if proc.returncode not in (0, 1, 5):
            tail = stdout[-OUTPUT_TAIL_CHARS:]
            raise TestRunError(f"pytest collection failed (exit {proc.returncode})\n{tail}")
        self.collection_errors = {
            path
            for line in stdout.splitlines()
            if line.startswith("ERROR ") and (path := _collection_error_path(line))
        }
        for path in sorted(self.collection_errors):
            log.warning("[tests] %s fails to import (collection error)", path)
        return [
            line.strip()
            for line in stdout.splitlines()
            if "::" in line and not line.startswith("ERROR")
        ]

    def _unknown_ids(self, requested: Sequence[str], timeout: float | None = None) -> set[str]:
        """Requested ids pytest does not collect.

        pytest refuses to run *any* test when one requested id doesn't exist, so unknown ids
        are filtered out up front. If collection itself fails (e.g. an import error in a test
        file, which may be the very bug to fix), nothing is filtered and pytest reports it.
        """
        try:
            collected = set(self.collect(timeout))
        except TestRunError as exc:
            log.debug("Collection failed; running requested ids unfiltered: %s", exc)
            self._collected = set()
            return set()
        self._collected = collected
        unknown = set()
        for nid in requested:
            known = (
                nid in collected
                or _file_of(nid) in self.collection_errors
                or ("[" not in nid and any(c.startswith(nid + "[") for c in collected))
            )
            if not known:
                unknown.add(nid)
                log.warning("[tests] %s is not collected by pytest", nid)
        return unknown

    def _was_collected(self, nid: str) -> bool:
        if nid in self._collected:
            return True
        return "[" not in nid and any(c.startswith(nid + "[") for c in self._collected)

    def run(
        self, node_ids: Sequence[str], extra_args: Sequence[str] = (), timeout: float | None = None
    ) -> TestRunResult:
        """Run ``node_ids`` and return results keyed by the requested ids.

        ``extra_args`` are added after the runner's own (for this run only, not collection);
        ``timeout`` overrides the runner's timeout for both collection and the run.
        """
        requested = list(dict.fromkeys(node_ids))
        unknown = self._unknown_ids(requested, timeout)
        runnable = [nid for nid in requested if nid not in unknown]
        if not runnable:
            log.info("[tests] None of the %d requested test(s) were collected", len(requested))
            results = {nid: _not_found(nid) for nid in requested}
            return TestRunResult(results=results, exit_code=5, duration=0.0, output_tail="")
        self._runs += 1
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        junit = self.reports_dir / f"run-{self._runs}.xml"
        junit.unlink(missing_ok=True)
        targets = dict.fromkeys(
            # A file that fails to import must be passed as a path, not a node id, or pytest
            # aborts the whole run; the collection error is then mapped back to each id.
            _file_of(nid) if _file_of(nid) in self.collection_errors else nid
            for nid in runnable
        )
        log.info("[tests] Running %d test(s)", len(runnable))
        started = time.monotonic()
        proc = self._pytest(
            *targets, *self._junit_args(junit), *self.extra_args, *extra_args, timeout=timeout
        )
        duration = time.monotonic() - started
        output_tail = (proc.stdout or "")[-OUTPUT_TAIL_CHARS:]

        cases: list[TestResult] = []
        parsed = False
        if junit.is_file() and junit.stat().st_size > 0:
            try:
                cases = _parse_junit(junit)
                parsed = True
            except ET.ParseError as exc:
                log.debug("Could not parse %s: %s", junit, exc)
        # Usage errors (exit 4, e.g. a path that doesn't exist) still write an empty report.
        if parsed and not cases and proc.returncode not in (0, 5):
            parsed = False

        results: dict[str, TestResult] = {nid: _not_found(nid) for nid in unknown}
        if not parsed:
            for nid in runnable:
                results[nid] = TestResult(
                    node_id=nid,
                    status=TestStatus.ERROR,
                    message=f"pytest did not produce results (exit {proc.returncode})",
                    details=output_tail,
                )
        else:
            by_id = {c.node_id: c for c in cases}
            for nid in runnable:
                if nid in by_id:
                    results[nid] = by_id[nid].model_copy(update={"node_id": nid})
                    continue
                matches = (
                    [c for c in cases if c.node_id.startswith(nid + "[")] if "[" not in nid else []
                )
                file_error = by_id.get(_file_of(nid))
                if matches:
                    results[nid] = _aggregate(nid, matches)
                elif file_error is not None and file_error.status is TestStatus.ERROR:
                    results[nid] = file_error.model_copy(update={"node_id": nid})
                elif self._was_collected(nid):
                    results[nid] = _not_found(nid, "not run (pytest stopped early)")
                else:
                    results[nid] = _not_found(nid)
        results = {nid: results[nid] for nid in requested}  # keep request order

        for nid, res in results.items():
            log.debug("[tests] %s: %s", nid, res.status.value)
        counts = {s: sum(1 for r in results.values() if r.status is s) for s in TestStatus}
        log.info(
            "[tests] %d passed, %d failed, %d error, %d not found in %.1fs",
            counts[TestStatus.PASSED],
            counts[TestStatus.FAILED],
            counts[TestStatus.ERROR],
            counts[TestStatus.NOT_FOUND],
            duration,
        )
        return TestRunResult(
            results=results, exit_code=proc.returncode, duration=duration, output_tail=output_tail
        )


def resolve_test_names(names: Sequence[str], collected: Sequence[str]) -> ResolvedTests:
    """Map user-supplied test names (full ids, bare names, ``Cls::name``) to collected node ids."""
    node_ids: list[str] = []
    ambiguous: dict[str, list[str]] = {}
    not_found: list[str] = []
    collected_set = set(collected)

    for name in names:
        if ".py" in name:
            if name in collected_set or any(c.startswith(name + "[") for c in collected):
                if name not in node_ids:
                    node_ids.append(name)
            else:
                log.warning("Test %s was not collected by pytest", name)
                not_found.append(name)
            continue

        suffix = "::" + name
        bases: set[str] = set()
        for cid in collected:
            stripped = _strip_param(cid)
            if cid.endswith(suffix):
                # Name with an explicit [param] matches only that case.
                bases.add(cid if "[" in name else stripped)
            elif "[" not in name and stripped.endswith(suffix):
                bases.add(stripped)
        if len(bases) == 1:
            resolved = bases.pop()
            if resolved not in node_ids:
                node_ids.append(resolved)
        elif len(bases) > 1:
            ambiguous[name] = sorted(bases)
            log.warning("Test name %s is ambiguous: %s", name, ", ".join(sorted(bases)))
        else:
            log.warning("Test %s was not collected by pytest", name)
            not_found.append(name)

    return ResolvedTests(node_ids=node_ids, ambiguous=ambiguous, not_found=not_found)
