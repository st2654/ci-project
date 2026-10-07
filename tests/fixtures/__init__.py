"""Shared test fixtures data. ``sample_repo/`` is a tiny project with seeded bugs."""

from __future__ import annotations

import shutil
from pathlib import Path

SAMPLE_REPO = Path(__file__).parent / "sample_repo"


def copy_sample_repo(dest: Path) -> Path:
    """Copy the sample repo to ``dest`` (which must not exist) and return ``dest``."""
    shutil.copytree(
        SAMPLE_REPO, dest, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache")
    )
    return dest
