"""Slice 6: ``PytestRunner.run_all`` (full suite) and ``GitRepo.stash_all`` / ``unstash``."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from conftest import git
from fixtures import copy_sample_repo

from ci_fix.tools.git import GitError, GitRepo
from ci_fix.tools.pytest_runner import PytestRunner, TestRunError, TestRunResult, TestStatus

OPS = "tests/test_ops.py"
ERRS = "tests/test_errors.py"
EXPECTED = {
    f"{OPS}::test_add": TestStatus.PASSED,
    f"{OPS}::test_subtract": TestStatus.FAILED,
    f"{OPS}::TestDivide::test_divide_ok": TestStatus.PASSED,
    f"{OPS}::TestDivide::test_divide_by_zero": TestStatus.FAILED,
    f"{OPS}::test_mean[xs0-2]": TestStatus.FAILED,
    f"{OPS}::test_mean[xs1-15]": TestStatus.FAILED,
    f"{OPS}::test_skipped": TestStatus.SKIPPED,
    f"{ERRS}::test_fixture_error": TestStatus.ERROR,
    f"{ERRS}::test_add": TestStatus.PASSED,
}


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    return copy_sample_repo(tmp_path / "sample_repo")


@pytest.fixture
def reports(tmp_path: Path) -> Path:
    path = tmp_path / "reports"
    path.mkdir()
    return path


@pytest.fixture
def runner(sample_repo: Path, reports: Path) -> PytestRunner:
    return PytestRunner(sample_repo, Path(sys.executable), reports)


@pytest.fixture(scope="module")
def full(tmp_path_factory: pytest.TempPathFactory) -> tuple[TestRunResult, Path]:
    base = tmp_path_factory.mktemp("run_all")
    repo = copy_sample_repo(base / "sample_repo")
    (base / "reports").mkdir()
    result = PytestRunner(repo, Path(sys.executable), base / "reports").run_all()
    return result, base / "reports"


# ---- run_all --------------------------------------------------------------------------------


def test_run_all_returns_every_sample_test_with_its_status(full) -> None:
    result, _ = full
    assert {nid: r.status for nid, r in result.results.items()} == EXPECTED


def test_run_all_results_keyed_by_real_node_id(full) -> None:
    result, _ = full
    assert all(r.node_id == nid for nid, r in result.results.items())


def test_run_all_writes_full_report(full) -> None:
    result, reports = full
    assert result.exit_code != 0
    assert (reports / "full-1.xml").is_file()
    ET.parse(reports / "full-1.xml")  # valid JUnit XML
    assert not list(reports.glob("run-*.xml"))


def test_run_all_report_numbering(runner: PytestRunner, reports: Path) -> None:
    runner.run_all(extra_args=["-k", "divide_ok"])
    runner.run_all(extra_args=["-k", "divide_ok"])
    assert (reports / "full-1.xml").is_file()
    assert (reports / "full-2.xml").is_file()


def test_run_all_respects_extra_args(runner: PytestRunner) -> None:
    result = runner.run_all(extra_args=["-k", "subtract"])
    assert list(result.results) == [f"{OPS}::test_subtract"]
    assert result.results[f"{OPS}::test_subtract"].status == TestStatus.FAILED


def test_run_all_all_passing_exit_code_zero(runner: PytestRunner) -> None:
    result = runner.run_all(extra_args=["-k", "divide_ok"])
    assert result.exit_code == 0
    assert result.results[f"{OPS}::TestDivide::test_divide_ok"].status == TestStatus.PASSED


def test_run_all_collection_error_keyed_by_path(runner: PytestRunner, sample_repo: Path) -> None:
    (sample_repo / "tests" / "test_broken.py").write_text("def test_x(:\n    pass\n")
    result = runner.run_all()
    broken = result.results["tests/test_broken.py"]
    assert broken.status == TestStatus.ERROR
    assert broken.node_id == "tests/test_broken.py"


def test_run_all_collection_error_does_not_hide_other_tests(
    runner: PytestRunner, sample_repo: Path
) -> None:
    """Otherwise every passing test would look 'missing' (= regressed) after the error."""
    (sample_repo / "tests" / "test_broken.py").write_text("import does_not_exist_xyz\n")
    result = runner.run_all()
    assert result.results["tests/test_broken.py"].status == TestStatus.ERROR
    assert result.results[f"{OPS}::test_add"].status == TestStatus.PASSED
    assert result.results[f"{ERRS}::test_add"].status == TestStatus.PASSED


def test_run_all_timeout_raises(runner: PytestRunner, sample_repo: Path) -> None:
    (sample_repo / "tests" / "test_slow.py").write_text(
        "import time\n\n\ndef test_slow():\n    time.sleep(30)\n"
    )
    with pytest.raises(TestRunError):
        runner.run_all(timeout=1)


def test_run_all_finds_new_test_file(runner: PytestRunner, sample_repo: Path) -> None:
    (sample_repo / "tests" / "test_new.py").write_text("def test_new():\n    assert True\n")
    result = runner.run_all()
    assert result.results["tests/test_new.py::test_new"].status == TestStatus.PASSED


# ---- stash_all / unstash ---------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path) -> GitRepo:
    path = tmp_path / "repo"
    path.mkdir()
    git("init", "-q", "-b", "main", cwd=path)
    git("config", "user.email", "tester@example.com", cwd=path)
    git("config", "user.name", "Tester", cwd=path)
    git("config", "commit.gpgsign", "false", cwd=path)
    (path / ".gitignore").write_text("*.log\n")
    (path / "app.py").write_text("x = 1\n")
    (path / "other.py").write_text("y = 1\n")
    git("add", "-A", cwd=path)
    git("commit", "-q", "-m", "initial", cwd=path)
    return GitRepo(path)


def test_stash_all_nothing_to_stash_returns_false(repo: GitRepo) -> None:
    assert repo.stash_all() is False
    assert git("stash", "list", cwd=repo.path) == ""


def test_stash_all_ignored_only_returns_false(repo: GitRepo) -> None:
    (repo.path / "build.log").write_text("log\n")
    assert repo.stash_all() is False
    assert (repo.path / "build.log").read_text() == "log\n"


def test_stash_and_unstash_round_trip(repo: GitRepo) -> None:
    (repo.path / "app.py").write_text("x = 2\n")
    (repo.path / "pkg").mkdir()
    (repo.path / "pkg" / "new.py").write_text("z = 3\n")
    (repo.path / "build.log").write_text("ignored\n")
    head = git("rev-parse", "HEAD", cwd=repo.path)

    assert repo.stash_all() is True
    assert (repo.path / "app.py").read_text() == "x = 1\n"
    assert not (repo.path / "pkg" / "new.py").exists()
    assert git("status", "--porcelain", cwd=repo.path) == ""
    assert (repo.path / "build.log").read_text() == "ignored\n"  # ignored files untouched

    repo.unstash()
    assert (repo.path / "app.py").read_text() == "x = 2\n"
    assert (repo.path / "pkg" / "new.py").read_text() == "z = 3\n"
    assert (repo.path / "other.py").read_text() == "y = 1\n"
    assert (repo.path / "build.log").read_text() == "ignored\n"
    assert git("rev-parse", "HEAD", cwd=repo.path) == head
    assert git("stash", "list", cwd=repo.path) == ""
    assert "pkg/new.py" in git("status", "--porcelain", "-uall", cwd=repo.path)


def test_stash_deleted_file_restored(repo: GitRepo) -> None:
    (repo.path / "other.py").unlink()
    assert repo.stash_all() is True
    assert (repo.path / "other.py").read_text() == "y = 1\n"
    repo.unstash()
    assert not (repo.path / "other.py").exists()


def test_unstash_keeps_index_state(repo: GitRepo) -> None:
    (repo.path / "app.py").write_text("x = 2\n")
    git("add", "app.py", cwd=repo.path)
    assert repo.stash_all() is True
    repo.unstash()
    assert git("diff", "--cached", "--name-only", cwd=repo.path) == "app.py"


def test_unstash_conflict_raises_git_error(repo: GitRepo) -> None:
    (repo.path / "app.py").write_text("x = 2\n")
    assert repo.stash_all() is True
    (repo.path / "app.py").write_text("x = 3\n")
    git("commit", "-q", "-am", "conflicting", cwd=repo.path)
    with pytest.raises(GitError):
        repo.unstash()


def test_unstash_without_stash_raises_git_error(repo: GitRepo) -> None:
    with pytest.raises(GitError):
        repo.unstash()


def test_stash_round_trip_after_diff_with_intent_to_add(repo: GitRepo) -> None:
    (repo.path / "app.py").write_text("x = 2\n")
    (repo.path / "new.py").write_text("n = 1\n")
    repo.diff("HEAD")  # leaves intent-to-add entries, which plain `git stash` rejects
    before = repo.changed_files()
    assert before == ["app.py", "new.py"]

    assert repo.stash_all() is True
    assert git("status", "--porcelain", cwd=repo.path) == ""
    assert not (repo.path / "new.py").exists()

    repo.unstash()
    assert repo.changed_files() == before
    assert (repo.path / "new.py").read_text() == "n = 1\n"
    assert (repo.path / "app.py").read_text() == "x = 2\n"


def test_added_files_lists_only_new_paths(repo: GitRepo) -> None:
    (repo.path / "app.py").write_text("x = 2\n")
    (repo.path / "pkg").mkdir()
    (repo.path / "pkg" / "new.py").write_text("z = 3\n")
    (repo.path / "build.log").write_text("ignored\n")
    assert repo.added_files() == {"pkg/new.py"}
    assert repo.added_files() == {"pkg/new.py"}  # stable once intent-to-add is set
