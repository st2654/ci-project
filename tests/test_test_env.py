"""Tests for ci_fix.tools.test_env.create_test_env.

Unit tests replace ``run_target`` with a recorder, so no venv is created and nothing is
installed. One opt-in integration test (CI_FIX_RUN_INTEGRATION=1) builds a real env.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from fixtures import copy_sample_repo

from ci_fix.config import Settings
from ci_fix.tools import test_env as test_env_module
from ci_fix.tools.pytest_runner import PytestRunner, TestStatus
from ci_fix.tools.target_env import build_target_env
from ci_fix.tools.test_env import TestEnv, TestEnvError, create_test_env

Argv = list[str]


class Recorder:
    """Stands in for target_env.run_target: records argv, returns rc=1 where ``fails(argv)``."""

    def __init__(self, fails: Callable[[Argv], bool] = lambda argv: False) -> None:
        self.calls: list[Argv] = []
        self.fails = fails

    def __call__(self, args: Sequence[str], *a: object, **kw: object):
        argv = [str(x) for x in args]
        self.calls.append(argv)
        rc = 1 if self.fails(argv) else 0
        return subprocess.CompletedProcess(argv, rc, stdout="out\n", stderr="")

    def find(self, pred: Callable[[Argv], bool]) -> list[Argv]:
        return [c for c in self.calls if pred(c)]


def _is_editable(argv: Argv) -> bool:
    return "-e" in argv


def _is_pytest_check(argv: Argv) -> bool:
    return argv[-2:] == ["-c", "import pytest"]


def _is_pytest_install(argv: Argv) -> bool:
    return "install" in argv and argv[-1] == "pytest" and "-e" not in argv


def _install_calls(rec: Recorder) -> list[Argv]:
    """Every call between creating the venv and the pytest import check."""
    return [c for c in rec.calls[1:] if not _is_pytest_check(c)]


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    rec = Recorder()
    monkeypatch.setattr(test_env_module, "run_target", rec)
    return rec


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "repo"
    path.mkdir()
    return path


@pytest.fixture
def venv(tmp_path: Path) -> Path:
    return tmp_path / "venv"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(workspace_dir=tmp_path / "ws")


def _use(monkeypatch: pytest.MonkeyPatch, rec: Recorder) -> Recorder:
    monkeypatch.setattr(test_env_module, "run_target", rec)
    return rec


# ---- python version -----------------------------------------------------------------------


def test_python_version_from_settings(
    recorder: Recorder, repo: Path, venv: Path, settings: Settings
) -> None:
    env = create_test_env(repo, venv, settings)
    assert recorder.calls[0][:4] == ["uv", "venv", "--python", "3.11"]
    assert str(venv) in recorder.calls[0]
    assert isinstance(env, TestEnv)
    assert env.venv_dir == venv
    assert env.python == venv / "bin" / "python"


def test_python_version_from_file(recorder: Recorder, repo: Path, venv: Path) -> None:
    (repo / ".python-version").write_text("3.12\n")
    create_test_env(repo, venv, Settings(python_version="3.13"))
    assert recorder.calls[0][:4] == ["uv", "venv", "--python", "3.12"]


def test_python_version_setting_used_when_file_blank(
    recorder: Recorder, repo: Path, venv: Path
) -> None:
    (repo / ".python-version").write_text("\n")
    create_test_env(repo, venv, Settings(python_version="3.13"))
    assert recorder.calls[0][:4] == ["uv", "venv", "--python", "3.13"]


# ---- install method precedence ------------------------------------------------------------


def test_custom_install_command_wins(recorder: Recorder, repo: Path, venv: Path) -> None:
    (repo / "uv.lock").write_text("")
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n")
    (repo / "requirements.txt").write_text("requests\n")
    cmd = ["make", "install-test-deps"]
    env = create_test_env(repo, venv, Settings(install_command=cmd))
    assert env.install_method == "custom"
    assert _install_calls(recorder) == [cmd]


def test_uv_lock_uses_uv_sync(recorder: Recorder, repo: Path, venv: Path) -> None:
    (repo / "uv.lock").write_text("")
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n")
    env = create_test_env(repo, venv, Settings())
    assert env.install_method == "uv-sync"
    installs = _install_calls(recorder)
    assert len(installs) == 1
    assert installs[0][:2] == ["uv", "sync"]


@pytest.mark.parametrize("marker", ["pyproject.toml", "setup.py"])
def test_project_file_uses_editable_test_extra(
    recorder: Recorder, repo: Path, venv: Path, marker: str
) -> None:
    (repo / marker).write_text("")
    (repo / "requirements.txt").write_text("requests\n")  # lower precedence: ignored
    env = create_test_env(repo, venv, Settings())
    assert env.install_method == "editable"
    installs = _install_calls(recorder)
    assert len(installs) == 1
    assert installs[0][-2:] == ["-e", ".[test]"]
    assert installs[0][:3] == ["uv", "pip", "install"]
    assert str(venv / "bin" / "python") in installs[0]


def test_editable_falls_back_to_dev_extra(
    monkeypatch: pytest.MonkeyPatch, repo: Path, venv: Path
) -> None:
    rec = _use(monkeypatch, Recorder(fails=lambda argv: ".[test]" in argv))
    (repo / "pyproject.toml").write_text("")
    env = create_test_env(repo, venv, Settings())
    assert env.install_method == "editable"
    assert [c[-1] for c in rec.find(_is_editable)] == [".[test]", ".[dev]"]


def test_editable_falls_back_to_plain(
    monkeypatch: pytest.MonkeyPatch, repo: Path, venv: Path
) -> None:
    rec = _use(monkeypatch, Recorder(fails=lambda argv: ".[test]" in argv or ".[dev]" in argv))
    (repo / "pyproject.toml").write_text("")
    create_test_env(repo, venv, Settings())
    assert [c[-1] for c in rec.find(_is_editable)] == [".[test]", ".[dev]", "."]


def test_editable_all_fail_raises(monkeypatch: pytest.MonkeyPatch, repo: Path, venv: Path) -> None:
    rec = _use(monkeypatch, Recorder(fails=_is_editable))
    (repo / "pyproject.toml").write_text("")
    with pytest.raises(TestEnvError):
        create_test_env(repo, venv, Settings())
    assert len(rec.find(_is_editable)) == 3


def test_requirements_files(recorder: Recorder, repo: Path, venv: Path) -> None:
    (repo / "requirements.txt").write_text("requests\n")
    (repo / "requirements-dev.txt").write_text("pytest\n")
    env = create_test_env(repo, venv, Settings())
    assert env.install_method == "requirements"
    installs = _install_calls(recorder)
    assert len(installs) == 1
    argv = installs[0]
    assert argv[:3] == ["uv", "pip", "install"]
    for name in ("requirements.txt", "requirements-dev.txt"):
        assert argv[argv.index(name) - 1] == "-r"


def test_no_project_files_installs_nothing(recorder: Recorder, repo: Path, venv: Path) -> None:
    env = create_test_env(repo, venv, Settings())
    assert env.install_method == "none"
    assert _install_calls(recorder) == []
    assert len(recorder.find(_is_pytest_check)) == 1


def test_commands_run_in_repo(monkeypatch: pytest.MonkeyPatch, repo: Path, venv: Path) -> None:
    cwds: list[object] = []

    def fake_run(args, *a, **kw):
        cwds.append(kw.get("cwd"))
        return subprocess.CompletedProcess(list(args), 0, stdout="", stderr="")

    monkeypatch.setattr(test_env_module, "run_target", fake_run)
    (repo / "pyproject.toml").write_text("")
    create_test_env(repo, venv, Settings())
    assert cwds and all(Path(str(c)) == repo for c in cwds)


# ---- failures & pytest check --------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "fails"),
    [
        ("venv", lambda argv: argv[:2] == ["uv", "venv"]),
        ("uv.lock", lambda argv: argv[:2] == ["uv", "sync"]),
        ("requirements.txt", lambda argv: "-r" in argv),
        ("custom", lambda argv: argv == ["make", "deps"]),
    ],
)
def test_failing_step_raises(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    venv: Path,
    shape: str,
    fails: Callable[[Argv], bool],
) -> None:
    _use(monkeypatch, Recorder(fails=fails))
    settings = Settings(install_command=["make", "deps"]) if shape == "custom" else Settings()
    if shape not in ("venv", "custom"):
        (repo / shape).write_text("")
    with pytest.raises(TestEnvError):
        create_test_env(repo, venv, settings)


def test_pytest_already_importable_not_installed(
    recorder: Recorder, repo: Path, venv: Path
) -> None:
    (repo / "pyproject.toml").write_text("")
    create_test_env(repo, venv, Settings())
    checks = recorder.find(_is_pytest_check)
    assert len(checks) == 1
    assert checks[0][0] == str(venv / "bin" / "python")
    assert recorder.find(_is_pytest_install) == []


def test_pytest_installed_when_import_fails(
    monkeypatch: pytest.MonkeyPatch, repo: Path, venv: Path
) -> None:
    rec = _use(monkeypatch, Recorder(fails=_is_pytest_check))
    (repo / "pyproject.toml").write_text("")
    create_test_env(repo, venv, Settings())
    installs = rec.find(_is_pytest_install)
    assert len(installs) == 1
    assert rec.calls.index(installs[0]) > rec.calls.index(rec.find(_is_pytest_check)[0])


def test_pytest_install_failure_raises(
    monkeypatch: pytest.MonkeyPatch, repo: Path, venv: Path
) -> None:
    _use(monkeypatch, Recorder(fails=lambda a: _is_pytest_check(a) or _is_pytest_install(a)))
    with pytest.raises(TestEnvError):
        create_test_env(repo, venv, Settings())


def test_command_not_found_raises(monkeypatch: pytest.MonkeyPatch, repo: Path, venv: Path) -> None:
    def boom(*a, **kw):
        raise FileNotFoundError("uv")

    monkeypatch.setattr(test_env_module, "run_target", boom)
    with pytest.raises(TestEnvError):
        create_test_env(repo, venv, Settings())


def test_timeout_raises(monkeypatch: pytest.MonkeyPatch, repo: Path, venv: Path) -> None:
    def slow(args, *a, **kw):
        raise subprocess.TimeoutExpired(args, kw.get("timeout") or 1)

    monkeypatch.setattr(test_env_module, "run_target", slow)
    with pytest.raises(TestEnvError):
        create_test_env(repo, venv, Settings(test_timeout_seconds=1))


# ---- integration --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("CI_FIX_RUN_INTEGRATION") != "1",
    reason="set CI_FIX_RUN_INTEGRATION=1 to build a real virtualenv (needs uv + network)",
)
def test_real_env_runs_sample_repo(tmp_path: Path) -> None:
    repo = copy_sample_repo(tmp_path / "repo")
    env = create_test_env(repo, tmp_path / "venv", Settings(workspace_dir=tmp_path / "ws"))
    assert env.install_method == "editable"
    assert env.python.exists()

    reports = tmp_path / "reports"
    reports.mkdir()
    add = "tests/test_ops.py::test_add"
    sub = "tests/test_ops.py::test_subtract"
    result = PytestRunner(repo, env.python, reports).run([add, sub])
    assert result.results[add].status == TestStatus.PASSED
    assert result.results[sub].status == TestStatus.FAILED


# ---- target environment isolation ---------------------------------------------------------


def test_build_target_env_drops_secrets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in ("GITHUB_TOKEN", "ANTHROPIC_API_KEY", "PYTHONPATH", "CI_FIX_FOO", "PYTHONHOME"):
        monkeypatch.setenv(key, "x")
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:3128")
    plugin = tmp_path / "plugin"
    env = build_target_env(tmp_path / "venv", [plugin])
    for key in ("GITHUB_TOKEN", "ANTHROPIC_API_KEY", "CI_FIX_FOO", "PYTHONHOME"):
        assert key not in env
    assert env["PYTHONPATH"] == str(plugin)
    assert env["LC_ALL"] == "C.UTF-8"
    assert env["HTTPS_PROXY"] == "http://proxy:3128"
    assert env["VIRTUAL_ENV"] == str(tmp_path / "venv")
    assert env["UV_PROJECT_ENVIRONMENT"] == str(tmp_path / "venv")
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PATH"].split(os.pathsep)[0] == str(tmp_path / "venv" / "bin")


def test_build_target_env_no_pythonpath_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/leak")
    assert "PYTHONPATH" not in build_target_env(tmp_path / "venv")


def test_build_target_env_strips_own_venv_from_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    own_bin = str(Path(sys.prefix) / "bin")
    monkeypatch.setenv("PATH", os.pathsep.join([own_bin, "/usr/bin"]))
    path = build_target_env(tmp_path / "venv")["PATH"].split(os.pathsep)
    assert own_bin not in path
    assert "/usr/bin" in path


def test_create_test_env_uses_isolated_env(
    monkeypatch: pytest.MonkeyPatch, repo: Path, venv: Path
) -> None:
    envs: list[dict[str, str]] = []

    def fake_run(args, *a, **kw):
        envs.append(dict(kw["env"]))
        return subprocess.CompletedProcess(list(args), 0, stdout="", stderr="")

    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setattr(test_env_module, "run_target", fake_run)
    create_test_env(repo, venv, Settings())
    assert envs and all("GITHUB_TOKEN" not in e for e in envs)
    assert all(e["VIRTUAL_ENV"] == str(venv) for e in envs)


def test_env_command_timeout_kills_process_tree(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    script = f"sleep 300 & echo $! > {pid_file}; sleep 300"
    with pytest.raises(TestEnvError):
        test_env_module._run(["sh", "-c", script], tmp_path, build_target_env(tmp_path), 2)
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        stat = Path(f"/proc/{pid}/stat")
        if stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"child process {pid} survived the timeout")
