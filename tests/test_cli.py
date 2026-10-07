"""Tests for ci_fix.cli (slice 0)."""

from __future__ import annotations

import pytest

from ci_fix.cli import main


def test_version_flag_prints_version_and_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert out.strip() == "ci-fix 0.1.0"


def test_no_args_prints_not_implemented_and_returns_nonzero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main([])
    assert rc == 2
    captured = capsys.readouterr()
    assert "not implemented" in (captured.out + captured.err).lower()


def test_main_none_argv_uses_sys_argv(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["ci-fix"])
    assert main(None) == 2


def test_unknown_flag_exits_nonzero() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--definitely-not-a-flag"])
    assert exc.value.code != 0
