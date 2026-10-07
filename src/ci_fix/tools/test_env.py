"""Create an isolated virtualenv for the target repo and install its test dependencies."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel

from ci_fix.config import Settings
from ci_fix.logging_setup import get_logger
from ci_fix.tools.target_env import build_target_env, run_target

log = get_logger(__name__)

OUTPUT_TAIL_CHARS = 2000
REQUIREMENTS_FILES = (
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-test.txt",
    "test-requirements.txt",
    "requirements/test.txt",
    "requirements/dev.txt",
)
EDITABLE_TARGETS = (".[test]", ".[dev]", ".")


class TestEnvError(Exception):
    """Raised when the test environment cannot be created or dependencies fail to install."""

    __test__ = False


class TestEnv(BaseModel):
    """A ready-to-use virtualenv for running the target repo's tests."""

    __test__ = False

    venv_dir: Path
    python: Path
    install_method: str


def _python_version(repo_path: Path, settings: Settings) -> str:
    version_file = repo_path / ".python-version"
    if version_file.is_file():
        lines = version_file.read_text(encoding="utf-8").splitlines()
        first = lines[0].strip() if lines else ""
        if first:
            return first
    return settings.python_version


def _run(
    argv: Sequence[str], cwd: Path, env: Mapping[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run a command; raise TestEnvError if it can't start or times out (not on non-zero exit)."""
    log.debug("$ %s (cwd=%s)", " ".join(argv), cwd)
    started = time.monotonic()
    try:
        proc = run_target(argv, cwd=cwd, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        raise TestEnvError(
            f"command timed out after {timeout}s: {' '.join(argv)}\n{out[-OUTPUT_TAIL_CHARS:]}"
        ) from None
    except OSError as exc:
        raise TestEnvError(f"failed to run {' '.join(argv)}: {exc}") from None
    log.debug("exit %d in %.2fs", proc.returncode, time.monotonic() - started)
    return proc


def _check(proc: subprocess.CompletedProcess[str]) -> None:
    if proc.returncode != 0:
        argv = proc.args if isinstance(proc.args, str) else " ".join(map(str, proc.args))
        tail = (proc.stdout or "")[-OUTPUT_TAIL_CHARS:]
        log.debug("command failed: %s\n%s", argv, tail)
        raise TestEnvError(f"command failed (exit {proc.returncode}): {argv}\n{tail}")


def create_test_env(repo_path: Path, venv_dir: Path, settings: Settings) -> TestEnv:
    """Create ``venv_dir`` with uv and install the repo's dependencies plus pytest."""
    started = time.monotonic()
    repo_path = Path(repo_path)
    venv_dir = Path(venv_dir)
    timeout = settings.test_timeout_seconds
    env = build_target_env(venv_dir)
    python = venv_dir / "bin" / "python"

    def run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return _run(argv, repo_path, env, timeout)

    version = _python_version(repo_path, settings)
    log.info("[env 1/3] Creating Python %s environment", version)
    _check(run(["uv", "venv", "--python", version, str(venv_dir)]))

    pip = ["uv", "pip", "install", "--python", str(python)]
    requirements = [name for name in REQUIREMENTS_FILES if (repo_path / name).is_file()]
    if settings.install_command:
        method = "custom"
    elif (repo_path / "uv.lock").is_file():
        method = "uv-sync"
    elif (repo_path / "pyproject.toml").is_file() or (repo_path / "setup.py").is_file():
        method = "editable"
    elif requirements:
        method = "requirements"
    else:
        method = "none"

    log.info("[env 2/3] Installing dependencies (%s)", method)
    if method == "custom":
        assert settings.install_command is not None
        _check(run(settings.install_command))
    elif method == "uv-sync":
        _check(run(["uv", "sync", "--frozen", "--all-extras"]))
    elif method == "editable":
        proc: subprocess.CompletedProcess[str] | None = None
        for target in EDITABLE_TARGETS:
            proc = run([*pip, "-e", target])
            if proc.returncode == 0:
                log.debug("Editable install succeeded with %s", target)
                break
            log.debug("Editable install with %s failed (exit %d)", target, proc.returncode)
        assert proc is not None
        _check(proc)
    elif method == "requirements":
        args: list[str] = []
        for name in requirements:
            args += ["-r", name]
        _check(run([*pip, *args]))

    log.info("[env 3/3] Ensuring pytest is available")
    if run([str(python), "-c", "import pytest"]).returncode != 0:
        _check(run([*pip, "pytest"]))

    log.info("Environment ready in %.1fs", time.monotonic() - started)
    return TestEnv(venv_dir=venv_dir, python=python, install_method=method)
