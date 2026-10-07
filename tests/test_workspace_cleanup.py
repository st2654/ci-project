"""Tests for slice 2 workspace layout (run_dir/repo, reports) and cleanup_workspace."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from conftest import REPO_URL, FakeRemote, fake_client, pr_info

from ci_fix import workspace as workspace_module
from ci_fix.config import Settings
from ci_fix.tools.github import RepoRef
from ci_fix.workspace import PreparedRepo, cleanup_workspace, prepare_pr_checkout


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(workspace_dir=tmp_path / "ws")


def _prepare(remote: FakeRemote, settings: Settings) -> PreparedRepo:
    return prepare_pr_checkout(
        REPO_URL,
        remote.pr_number,
        settings,
        fake_client(pr_info(remote)),
        clone_url=remote.url,
    )


def _prepared_at(remote: FakeRemote, run_dir: Path) -> PreparedRepo:
    return PreparedRepo(
        run_dir=run_dir,
        path=run_dir / "repo",
        repo=RepoRef(owner="octo", name="repo"),
        pr=pr_info(remote),
        branch=f"ci-fix/pr-{remote.pr_number}",
        pr_head_sha=remote.pr_sha,
    )


def test_layout(fake_remote: FakeRemote, settings: Settings) -> None:
    prepared = _prepare(fake_remote, settings)
    run_dir = Path(prepared.run_dir)
    assert run_dir.parent == settings.workspace_dir
    assert run_dir.name.startswith(f"octo__repo__pr-{fake_remote.pr_number}__")
    assert Path(prepared.path) == run_dir / "repo"
    assert prepared.venv_dir == run_dir / "venv"
    assert prepared.reports_dir == run_dir / "reports"
    assert (run_dir / "repo" / ".git").exists()
    assert (run_dir / "repo" / "app.py").is_file()
    assert (run_dir / "reports").is_dir()


def test_rerun_gets_empty_reports(fake_remote: FakeRemote, settings: Settings) -> None:
    first = _prepare(fake_remote, settings)
    (first.reports_dir / "run-1.xml").write_text("<testsuites/>")
    second = _prepare(fake_remote, settings)
    assert second.reports_dir.is_dir()
    assert list(second.reports_dir.iterdir()) == []


def test_cleanup_removes_run_dir(
    fake_remote: FakeRemote, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    prepared = _prepare(fake_remote, settings)
    caplog.set_level(logging.INFO, logger="ci_fix")
    cleanup_workspace(prepared, settings)
    assert not Path(prepared.run_dir).exists()
    assert settings.workspace_dir.is_dir()  # only the run dir goes, not the root
    assert any(
        r.levelno == logging.INFO and "Removed workspace" in r.getMessage() for r in caplog.records
    )


def test_cleanup_keep_workspace(
    fake_remote: FakeRemote, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = Settings(workspace_dir=tmp_path / "ws", keep_workspace=True)
    prepared = _prepare(fake_remote, settings)
    caplog.set_level(logging.INFO, logger="ci_fix")
    cleanup_workspace(prepared, settings)
    assert Path(prepared.path).is_dir()
    assert prepared.reports_dir.is_dir()
    assert any(
        r.levelno == logging.INFO and "Keeping workspace" in r.getMessage() for r in caplog.records
    )


def test_cleanup_refuses_outside_workspace(
    fake_remote: FakeRemote, settings: Settings, tmp_path: Path
) -> None:
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "repo").mkdir(parents=True)
    with pytest.raises(ValueError):
        cleanup_workspace(_prepared_at(fake_remote, elsewhere), settings)
    assert (elsewhere / "repo").is_dir()


def test_cleanup_refuses_workspace_root(fake_remote: FakeRemote, settings: Settings) -> None:
    settings.workspace_dir.mkdir(parents=True)
    marker = settings.workspace_dir / "other-run.txt"
    marker.write_text("keep me")
    with pytest.raises(ValueError):
        cleanup_workspace(_prepared_at(fake_remote, settings.workspace_dir), settings)
    assert marker.exists()


def test_cleanup_refuses_parent_traversal(
    fake_remote: FakeRemote, settings: Settings, tmp_path: Path
) -> None:
    settings.workspace_dir.mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.mkdir()
    with pytest.raises(ValueError):
        cleanup_workspace(
            _prepared_at(fake_remote, settings.workspace_dir / ".." / "victim"), settings
        )
    assert victim.is_dir()


def test_cleanup_refuses_symlink_escape(
    fake_remote: FakeRemote, settings: Settings, tmp_path: Path
) -> None:
    settings.workspace_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "data.txt").write_text("precious")
    link = settings.workspace_dir / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        cleanup_workspace(_prepared_at(fake_remote, link), settings)
    assert (outside / "data.txt").exists()


def test_cleanup_warns_when_removal_incomplete(
    fake_remote: FakeRemote,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    prepared = _prepare(fake_remote, settings)
    monkeypatch.setattr(workspace_module.shutil, "rmtree", lambda *a, **kw: None)
    caplog.set_level(logging.INFO, logger="ci_fix")
    cleanup_workspace(prepared, settings)
    assert Path(prepared.run_dir).exists()
    messages = [(r.levelno, r.getMessage()) for r in caplog.records]
    assert any(lvl == logging.WARNING and "Could not fully remove" in m for lvl, m in messages)
    assert not any("Removed workspace" in m for _, m in messages)
