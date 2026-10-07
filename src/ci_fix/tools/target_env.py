"""Environment and process helpers for running commands in the (untrusted) target repo.

The target repo's code comes from a pull request and must not see ci-fix's secrets, so its
commands get an allowlisted environment instead of a copy of ``os.environ``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

ALLOWED_ENV_KEYS = frozenset(
    {
        "HOME",
        "USER",
        "LOGNAME",
        "LANG",
        "LANGUAGE",
        "TERM",
        "TZ",
        "TMPDIR",
        "SHELL",
        # proxies / certificates
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        # package index configuration
        "PIP_INDEX_URL",
        "PIP_EXTRA_INDEX_URL",
        "UV_INDEX_URL",
        "UV_EXTRA_INDEX_URL",
        "UV_DEFAULT_INDEX",
        "UV_INDEX",
        # uv cache / managed pythons
        "UV_CACHE_DIR",
        "UV_PYTHON_INSTALL_DIR",
    }
)
ALLOWED_ENV_PREFIXES = ("LC_",)
_FALLBACK_PATH = "/usr/local/bin:/usr/bin:/bin"


def _is_inside(path: str, root: Path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(root)
    except (OSError, ValueError):
        return False


def _base_path() -> str:
    """The parent's PATH without entries inside ci-fix's own environment (``sys.prefix``)."""
    own = Path(sys.prefix).resolve()
    entries = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    kept = [p for p in entries if not _is_inside(p, own)]
    return os.pathsep.join(kept) or _FALLBACK_PATH


def build_target_env(venv_dir: Path, extra_pythonpath: Sequence[Path] = ()) -> dict[str, str]:
    """Build the environment for target-repo commands.

    Only allowlisted variables (locale, proxy/cert, package-index and uv cache settings) are
    copied from ``os.environ``; everything else — tokens, API keys, ``PYTHON*`` and
    ``CI_FIX_*`` variables — is dropped. ``venv_dir`` is activated and ``PYTHONPATH`` is set
    to ``extra_pythonpath`` only.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if key in ALLOWED_ENV_KEYS or key.startswith(ALLOWED_ENV_PREFIXES)
    }
    venv_dir = Path(venv_dir)
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["UV_PROJECT_ENVIRONMENT"] = str(venv_dir)
    env["PATH"] = os.pathsep.join([str(venv_dir / "bin"), _base_path()])
    if extra_pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(str(p) for p in extra_pythonpath)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def run_target(
    argv: Sequence[str], cwd: Path, env: Mapping[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    """Run ``argv`` in its own session, combining stdout and stderr.

    On timeout the whole process group (including any children the command spawned) is
    killed and ``subprocess.TimeoutExpired`` is raised with the output captured so far.
    ``OSError`` (e.g. command not found) propagates. A non-zero exit is not an error.
    """
    proc = subprocess.Popen(
        list(argv),
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, _ = proc.communicate()
        raise subprocess.TimeoutExpired(list(argv), timeout, output=stdout) from None
    return subprocess.CompletedProcess(list(argv), proc.returncode, stdout=stdout, stderr=None)
