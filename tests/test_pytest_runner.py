"""Tests for ci_fix.tools.pytest_runner against a copy of the sample repo (no installs)."""

from __future__ import annotations

import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from fixtures import copy_sample_repo

from ci_fix.tools import PytestRunner as ExportedRunner
from ci_fix.tools.pytest_runner import (
    PytestRunner,
    TestResult,
    TestRunError,
    TestRunResult,
    TestStatus,
)

OPS = "tests/test_ops.py"
ERRS = "tests/test_errors.py"
ALL_IDS = [
    f"{OPS}::test_add",
    f"{OPS}::test_subtract",
    f"{OPS}::TestDivide::test_divide_ok",
    f"{OPS}::TestDivide::test_divide_by_zero",
    f"{OPS}::test_mean[xs0-2]",
    f"{OPS}::test_mean[xs1-15]",
    f"{OPS}::test_skipped",
    f"{ERRS}::test_fixture_error",
    f"{ERRS}::test_add",
]


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    return copy_sample_repo(tmp_path / "sample_repo")


@pytest.fixture
def reports(tmp_path: Path) -> Path:
    return tmp_path / "reports"


@pytest.fixture
def runner(sample_repo: Path, reports: Path) -> PytestRunner:
    reports.mkdir()
    return PytestRunner(sample_repo, Path(sys.executable), reports)


@pytest.fixture(scope="module")
def full_run(tmp_path_factory: pytest.TempPathFactory) -> TestRunResult:
    """One run of every sample test, shared by the status checks (keeps the suite fast)."""
    base = tmp_path_factory.mktemp("full")
    repo = copy_sample_repo(base / "sample_repo")
    (base / "reports").mkdir()
    return PytestRunner(repo, Path(sys.executable), base / "reports").run(ALL_IDS)


def _ids(items: list) -> list[str]:
    """``failing``/``passed`` may hold node ids or TestResult objects; compare by id."""
    return [x if isinstance(x, str) else x.node_id for x in items]


# ---- models -------------------------------------------------------------------------------


def test_exported_from_tools_package() -> None:
    assert ExportedRunner is PytestRunner


def test_status_values() -> None:
    assert {s.value for s in TestStatus} == {"passed", "failed", "error", "skipped", "not_found"}


def test_result_defaults() -> None:
    r = TestResult(node_id="t.py::x", status=TestStatus.PASSED)
    assert (r.message, r.details, r.duration) == ("", "", 0.0)


def test_failing_and_passed_properties() -> None:
    results = {
        "a": TestResult(node_id="a", status=TestStatus.PASSED),
        "b": TestResult(node_id="b", status=TestStatus.ERROR),
        "c": TestResult(node_id="c", status=TestStatus.SKIPPED),
        "d": TestResult(node_id="d", status=TestStatus.FAILED),
        "e": TestResult(node_id="e", status=TestStatus.NOT_FOUND),
    }
    run = TestRunResult(results=results, exit_code=1, duration=0.1, output_tail="")
    assert _ids(run.failing) == ["b", "d"]
    assert _ids(run.passed) == ["a"]


# ---- collect ------------------------------------------------------------------------------


def test_collect(runner: PytestRunner) -> None:
    assert sorted(runner.collect()) == sorted(ALL_IDS)


def test_collect_records_broken_test_file(runner: PytestRunner, sample_repo: Path) -> None:
    (sample_repo / "tests" / "test_broken.py").write_text("def test_x(:\n    pass\n")
    ids = runner.collect()  # does not raise: the broken file may be the bug to fix
    assert runner.collection_errors == {"tests/test_broken.py"}
    assert f"{OPS}::test_add" in ids


def test_collect_bad_exit_raises(runner: PytestRunner, sample_repo: Path) -> None:
    (sample_repo / "tests" / "conftest.py").write_text("raise RuntimeError('broken conftest')\n")
    with pytest.raises(TestRunError):
        runner.collect()


# ---- run: statuses ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("node_id", "status"),
    [
        (f"{OPS}::test_add", TestStatus.PASSED),
        (f"{OPS}::test_subtract", TestStatus.FAILED),
        (f"{OPS}::TestDivide::test_divide_ok", TestStatus.PASSED),
        (f"{OPS}::TestDivide::test_divide_by_zero", TestStatus.FAILED),
        (f"{OPS}::test_mean[xs0-2]", TestStatus.FAILED),
        (f"{OPS}::test_mean[xs1-15]", TestStatus.FAILED),
        (f"{OPS}::test_skipped", TestStatus.SKIPPED),
        (f"{ERRS}::test_fixture_error", TestStatus.ERROR),
        (f"{ERRS}::test_add", TestStatus.PASSED),
    ],
)
def test_status_of_each_sample_test(
    full_run: TestRunResult, node_id: str, status: TestStatus
) -> None:
    assert full_run.results[node_id].status == status
    assert full_run.results[node_id].node_id == node_id


def test_results_keyed_by_requested_ids(full_run: TestRunResult) -> None:
    assert list(full_run.results) == ALL_IDS
    assert full_run.exit_code == 1
    assert full_run.duration > 0


def test_failing_in_request_order(full_run: TestRunResult) -> None:
    assert _ids(full_run.failing) == [
        f"{OPS}::test_subtract",
        f"{OPS}::TestDivide::test_divide_by_zero",
        f"{OPS}::test_mean[xs0-2]",
        f"{OPS}::test_mean[xs1-15]",
        f"{ERRS}::test_fixture_error",
    ]
    assert _ids(full_run.passed) == [
        f"{OPS}::test_add",
        f"{OPS}::TestDivide::test_divide_ok",
        f"{ERRS}::test_add",
    ]


def test_failure_message_and_details(full_run: TestRunResult) -> None:
    sub = full_run.results[f"{OPS}::test_subtract"]
    assert "assert" in sub.message
    assert "8 == 2" in sub.message or "8 == 2" in sub.details
    assert sub.details.strip()
    assert "test_ops.py" in sub.details

    div = full_run.results[f"{OPS}::TestDivide::test_divide_by_zero"]
    assert "ZeroDivisionError" in div.message + div.details
    assert div.details.strip()


def test_fixture_error_details(full_run: TestRunResult) -> None:
    err = full_run.results[f"{ERRS}::test_fixture_error"]
    assert "RuntimeError" in err.message + err.details
    assert "fixture setup failed" in err.message + err.details
    assert err.details.strip()


def test_passed_has_duration(full_run: TestRunResult) -> None:
    assert full_run.results[f"{OPS}::test_add"].duration >= 0.0


def test_output_tail_is_console_output(full_run: TestRunResult) -> None:
    assert isinstance(full_run.output_tail, str)
    assert "failed" in full_run.output_tail


# ---- run: id handling ---------------------------------------------------------------------


def test_parametrized_base_aggregates_to_failed(runner: PytestRunner) -> None:
    result = runner.run([f"{OPS}::test_mean"])
    assert list(result.results) == [f"{OPS}::test_mean"]
    assert result.results[f"{OPS}::test_mean"].status == TestStatus.FAILED


def test_exact_param_id(runner: PytestRunner) -> None:
    pid = f"{OPS}::test_mean[xs1-15]"
    result = runner.run([pid])
    assert list(result.results) == [pid]
    assert result.results[pid].status == TestStatus.FAILED


def _write(repo: Path, name: str, body: str) -> None:
    (repo / "tests" / name).write_text(body, encoding="utf-8")


PARAM_MODULE = """\
import pytest


@pytest.mark.parametrize("x", [1, 2])
def test_all_pass(x):
    assert x


@pytest.mark.parametrize("x", [1, 2])
def test_all_skip(x):
    pytest.skip("nope")


@pytest.mark.parametrize("x", [1, 2])
def test_one_fails(x):
    assert x == 1


@pytest.fixture(params=[1, 2])
def maybe_broken(request):
    if request.param == 2:
        raise RuntimeError("boom")
    return request.param


@pytest.mark.parametrize("y", [0])
def test_fail_and_error(maybe_broken, y):
    assert False


@pytest.mark.parametrize("x", [1, 2])
def test_pass_and_skip(x):
    if x == 2:
        pytest.skip("half")
"""


@pytest.mark.parametrize(
    ("name", "status"),
    [
        ("test_all_pass", TestStatus.PASSED),
        ("test_all_skip", TestStatus.SKIPPED),
        ("test_one_fails", TestStatus.FAILED),
        ("test_fail_and_error", TestStatus.ERROR),  # error wins over failure
        ("test_pass_and_skip", TestStatus.PASSED),
    ],
)
def test_parametrized_aggregation_rules(
    runner: PytestRunner, sample_repo: Path, name: str, status: TestStatus
) -> None:
    _write(sample_repo, "test_params.py", PARAM_MODULE)
    nid = f"tests/test_params.py::{name}"
    assert runner.run([nid]).results[nid].status == status


def test_unknown_test_in_existing_file_is_not_found(runner: PytestRunner) -> None:
    ids = [f"{OPS}::test_add", f"{OPS}::test_does_not_exist"]
    result = runner.run(ids)
    assert list(result.results) == ids
    assert result.results[f"{OPS}::test_does_not_exist"].status == TestStatus.NOT_FOUND
    assert result.results[f"{OPS}::test_add"].status == TestStatus.PASSED


def test_missing_file_is_not_found_and_others_still_run(runner: PytestRunner) -> None:
    ids = ["tests/test_nope.py::test_x", f"{OPS}::test_add"]
    result = runner.run(ids)  # must not raise
    assert list(result.results) == ids
    assert result.results["tests/test_nope.py::test_x"].status == TestStatus.NOT_FOUND
    assert result.results[f"{OPS}::test_add"].status == TestStatus.PASSED


def test_only_unknown_ids_skips_pytest_run(runner: PytestRunner, reports: Path) -> None:
    result = runner.run(["tests/test_nope.py::test_x"])
    assert result.results["tests/test_nope.py::test_x"].status == TestStatus.NOT_FOUND
    assert not list(reports.glob("run-*.xml"))


def test_collection_error_file_is_reported_not_hidden(sample_repo: Path, reports: Path) -> None:
    """A test file that fails to import may be the very bug to fix: report it as ERROR."""
    (sample_repo / "tests" / "test_broken_import.py").write_text(
        "from calc.ops import does_not_exist\n\n\n"
        "def test_uses_it():\n    assert does_not_exist()\n"
    )
    reports.mkdir()
    runner = PytestRunner(sample_repo, Path(sys.executable), reports)
    nid = "tests/test_broken_import.py::test_uses_it"
    result = runner.run([nid, f"{OPS}::test_add"])
    broken = result.results[nid]
    assert broken.status == TestStatus.ERROR
    assert "ImportError" in broken.message
    assert "does_not_exist" in broken.details
    assert nid in result.failing
    # Other tests still run despite the broken file.
    assert result.results[f"{OPS}::test_add"].status == TestStatus.PASSED
    assert runner.collection_errors == {"tests/test_broken_import.py"}


# ---- reports & timeout --------------------------------------------------------------------


def test_report_files_numbered_per_run(runner: PytestRunner, reports: Path) -> None:
    runner.run([f"{OPS}::test_add"])
    assert (reports / "run-1.xml").is_file()
    runner.run([f"{OPS}::test_subtract"])
    assert (reports / "run-2.xml").is_file()
    ET.parse(reports / "run-2.xml")  # valid JUnit XML
    names = {tc.get("name") for tc in ET.parse(reports / "run-2.xml").iter("testcase")}
    assert names == {"test_subtract"}


def test_extra_args_passed(sample_repo: Path, reports: Path) -> None:
    reports.mkdir()
    runner = PytestRunner(
        sample_repo, Path(sys.executable), reports, extra_args=["-p", "no:cacheprovider"]
    )
    result = runner.run([f"{OPS}::test_add"])
    assert result.results[f"{OPS}::test_add"].status == TestStatus.PASSED
    assert not (sample_repo / ".pytest_cache").exists()


def test_timeout_raises(sample_repo: Path, reports: Path) -> None:
    reports.mkdir()
    _write(sample_repo, "test_slow.py", "import time\n\n\ndef test_slow():\n    time.sleep(30)\n")
    runner = PytestRunner(sample_repo, Path(sys.executable), reports, timeout=1)
    with pytest.raises(TestRunError):
        runner.run(["tests/test_slow.py::test_slow"])


def _process_gone(pid: int, wait: float = 5.0) -> bool:
    """True once ``pid`` no longer exists (a zombie awaiting reaping counts as gone)."""
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        stat = Path(f"/proc/{pid}/stat")
        try:
            if stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return True
        except (OSError, IndexError):
            pass
        time.sleep(0.1)
    return False


def test_timeout_kills_process_tree(sample_repo: Path, reports: Path, tmp_path: Path) -> None:
    reports.mkdir()
    pid_file = tmp_path / "child.pid"
    _write(
        sample_repo,
        "test_spawns.py",
        "import subprocess, time\n\n\n"
        "def test_spawns():\n"
        '    child = subprocess.Popen(["sleep", "300"])\n'
        f"    open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "    time.sleep(300)\n",
    )
    runner = PytestRunner(sample_repo, Path(sys.executable), reports, timeout=3)
    with pytest.raises(TestRunError):
        runner.run(["tests/test_spawns.py::test_spawns"])
    pid = int(pid_file.read_text())
    assert _process_gone(pid), f"child process {pid} survived the timeout"


# ---- exact node ids -----------------------------------------------------------------------


def test_inherited_test_keeps_exact_node_id(runner: PytestRunner, sample_repo: Path) -> None:
    """A test defined on a base class in another file is reported under the child's id."""
    _write(
        sample_repo,
        "base_cases.py",
        "class AddCases:\n    def test_inherited(self):\n        assert 1 + 1 == 3\n",
    )
    _write(
        sample_repo,
        "test_child.py",
        "from base_cases import AddCases\n\n\nclass TestChild(AddCases):\n    pass\n",
    )
    nid = "tests/test_child.py::TestChild::test_inherited"
    result = runner.run([nid])
    assert result.results[nid].status == TestStatus.FAILED
    assert "base_cases.py" in result.results[nid].details


def test_teardown_error_after_failure_is_merged(runner: PytestRunner, sample_repo: Path) -> None:
    _write(
        sample_repo,
        "test_teardown.py",
        "import pytest\n\n\n"
        "@pytest.fixture\ndef bad_teardown():\n    yield\n"
        "    raise RuntimeError('teardown boom')\n\n\n"
        "def test_fails(bad_teardown):\n    assert 1 == 2\n",
    )
    nid = "tests/test_teardown.py::test_fails"
    res = runner.run([nid]).results[nid]
    assert res.status == TestStatus.ERROR
    assert "teardown boom" in res.details
    assert "assert 1 == 2" in res.details


# ---- environment isolation ----------------------------------------------------------------


def test_target_tests_do_not_see_secrets(
    runner: PytestRunner, sample_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "secret-token")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-key")
    _write(
        sample_repo,
        "test_secrets.py",
        "import os\n\n\ndef test_no_secrets():\n"
        '    assert "GITHUB_TOKEN" not in os.environ\n'
        '    assert "ANTHROPIC_API_KEY" not in os.environ\n',
    )
    runner = PytestRunner(sample_repo, Path(sys.executable), runner.reports_dir)
    nid = "tests/test_secrets.py::test_no_secrets"
    assert runner.run([nid]).results[nid].status == TestStatus.PASSED


def test_runner_default_env_is_isolated(
    sample_repo: Path, reports: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("PYTHONPATH", "/somewhere")
    runner = PytestRunner(sample_repo, Path(sys.executable), reports)
    assert "GITHUB_TOKEN" not in runner.env
    assert runner.env["PYTHONPATH"] == str(reports / "_plugin")


def test_runner_explicit_env_gets_plugin_dir(sample_repo: Path, reports: Path) -> None:
    runner = PytestRunner(
        sample_repo, Path(sys.executable), reports, env={"PATH": "/bin", "PYTHONPATH": "/a"}
    )
    assert runner.env["PYTHONPATH"] == os.pathsep.join(["/a", str(reports / "_plugin")])
    assert runner.env["PATH"] == "/bin"


# ---- extra args & stopping early ----------------------------------------------------------


def test_collect_uses_extra_args(sample_repo: Path, reports: Path) -> None:
    reports.mkdir()
    runner = PytestRunner(sample_repo, Path(sys.executable), reports, extra_args=["-k", "subtract"])
    assert runner.collect() == [f"{OPS}::test_subtract"]


def test_stopped_early_reported_as_not_run(sample_repo: Path, reports: Path) -> None:
    reports.mkdir()
    _write(
        sample_repo,
        "test_order.py",
        "def test_first():\n    assert False\n\n\ndef test_second():\n    pass\n",
    )
    runner = PytestRunner(sample_repo, Path(sys.executable), reports, extra_args=["-x"])
    first, second = "tests/test_order.py::test_first", "tests/test_order.py::test_second"
    result = runner.run([first, second])
    assert result.results[first].status == TestStatus.FAILED
    assert result.results[second].status == TestStatus.NOT_FOUND
    assert result.results[second].message == "not run (pytest stopped early)"


# ---- paths with spaces --------------------------------------------------------------------


def test_collection_error_path_with_space(runner: PytestRunner, sample_repo: Path) -> None:
    (sample_repo / "tests" / "test_broken file.py").write_text("def test_x(:\n    pass\n")
    runner.collect()
    assert runner.collection_errors == {"tests/test_broken file.py"}


def test_ids_stable_without_pytest_config(tmp_path: Path) -> None:
    """No pytest config, no setup.py: ids must still be repo-root relative in collect and run."""
    repo = tmp_path / "bare"
    (repo / "tests" / "unit").mkdir(parents=True)
    (repo / "tests" / "unit" / "test_m.py").write_text(
        "def test_ok():\n    assert True\n\n\ndef test_bad():\n    assert 1 == 2\n"
    )
    runner = PytestRunner(repo, Path(sys.executable), tmp_path / "reports")
    ids = runner.collect()
    assert "tests/unit/test_m.py::test_bad" in ids
    result = runner.run(["tests/unit/test_m.py::test_bad", "tests/unit/test_m.py::test_ok"])
    assert result.results["tests/unit/test_m.py::test_bad"].status == TestStatus.FAILED
    assert result.results["tests/unit/test_m.py::test_ok"].status == TestStatus.PASSED
