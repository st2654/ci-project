"""Tests for ci_fix.config (slice 0): Settings model and load_settings()."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

import ci_fix
from ci_fix import ConfigError, Settings, load_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_CONFIG = REPO_ROOT / "config.example.toml"

FAKE_ANTHROPIC_KEY = "sk-ant-test-0000-SUPERSECRET"
FAKE_GITHUB_TOKEN = "ghp_test_1111_SUPERSECRET"


def write_toml(tmp_path: Path, body: str, name: str = "config.toml") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Package surface
# --------------------------------------------------------------------------- #


def test_package_version() -> None:
    assert ci_fix.__version__ == "0.1.0"


def test_package_exports_config_api() -> None:
    from ci_fix import config

    assert ci_fix.Settings is config.Settings
    assert ci_fix.load_settings is config.load_settings
    assert ci_fix.ConfigError is config.ConfigError
    assert issubclass(ConfigError, Exception)


# --------------------------------------------------------------------------- #
# Settings model
# --------------------------------------------------------------------------- #


def test_settings_defaults() -> None:
    s = Settings()
    assert s.model == "claude-sonnet-4-6"
    assert s.temperature == 0.0
    assert s.max_attempts == 3
    assert s.max_parallel_workers == 4
    assert s.branch_prefix == "ci-fix/pr-"
    assert s.pytest_args == []
    assert s.regression_pytest_args == []
    assert s.regression_timeout_seconds == 1800
    assert s.anthropic_api_key is None
    assert s.github_token is None


def test_settings_workspace_dir_is_expanded() -> None:
    s = Settings()
    assert isinstance(s.workspace_dir, Path)
    assert "~" not in str(s.workspace_dir)
    assert s.workspace_dir == Path("~/.ci-fix/workspaces").expanduser()


def test_settings_is_frozen() -> None:
    s = Settings()
    with pytest.raises(ValidationError):
        s.max_attempts = 10  # type: ignore[misc]


def test_settings_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Settings(unknown_field=1)  # type: ignore[call-arg]


@pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0])
def test_settings_temperature_bounds_inclusive(temperature: float) -> None:
    assert Settings(temperature=temperature).temperature == temperature


def test_require_secrets_ok_when_both_set() -> None:
    s = Settings(
        anthropic_api_key=SecretStr(FAKE_ANTHROPIC_KEY),
        github_token=SecretStr(FAKE_GITHUB_TOKEN),
    )
    s.require_secrets()  # must not raise


@pytest.mark.parametrize(
    ("anthropic", "github", "missing"),
    [
        (None, None, ["ANTHROPIC_API_KEY", "GITHUB_TOKEN"]),
        (None, FAKE_GITHUB_TOKEN, ["ANTHROPIC_API_KEY"]),
        (FAKE_ANTHROPIC_KEY, None, ["GITHUB_TOKEN"]),
    ],
)
def test_require_secrets_names_every_missing_secret(
    anthropic: str | None, github: str | None, missing: list[str]
) -> None:
    s = Settings(
        anthropic_api_key=SecretStr(anthropic) if anthropic else None,
        github_token=SecretStr(github) if github else None,
    )
    with pytest.raises(ConfigError) as exc:
        s.require_secrets()
    msg = str(exc.value)
    for name in missing:
        assert name in msg
    for name in {"ANTHROPIC_API_KEY", "GITHUB_TOKEN"} - set(missing):
        assert name not in msg


def test_secrets_not_leaked_in_repr() -> None:
    s = Settings(
        anthropic_api_key=SecretStr(FAKE_ANTHROPIC_KEY),
        github_token=SecretStr(FAKE_GITHUB_TOKEN),
    )
    text = repr(s) + str(s)
    assert FAKE_ANTHROPIC_KEY not in text
    assert FAKE_GITHUB_TOKEN not in text


# --------------------------------------------------------------------------- #
# load_settings: file resolution
# --------------------------------------------------------------------------- #


def test_load_defaults_when_no_path_and_no_cwd_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert load_settings(env={}) == Settings()


def test_load_uses_cwd_config_toml_when_path_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_toml(tmp_path, "[ci_fix]\nmax_attempts = 7\n")
    monkeypatch.chdir(tmp_path)
    assert load_settings(env={}).max_attempts == 7


def test_load_explicit_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_settings(path=tmp_path / "does-not-exist.toml", env={})


def test_load_explicit_path_overrides_defaults(tmp_path: Path) -> None:
    path = write_toml(
        tmp_path,
        """
[ci_fix]
model = "claude-other"
temperature = 0.3
max_attempts = 5
max_parallel_workers = 2
workspace_dir = "~/somewhere/else"
branch_prefix = "fix/"
pytest_args = ["-x", "-q"]
regression_pytest_args = ["-m", "not slow"]
regression_timeout_seconds = 60
""",
        name="custom.toml",
    )
    s = load_settings(path=path, env={})
    assert s.model == "claude-other"
    assert s.temperature == 0.3
    assert s.max_attempts == 5
    assert s.max_parallel_workers == 2
    assert s.workspace_dir == Path("~/somewhere/else").expanduser()
    assert s.branch_prefix == "fix/"
    assert s.pytest_args == ["-x", "-q"]
    assert s.regression_pytest_args == ["-m", "not slow"]
    assert s.regression_timeout_seconds == 60


def test_load_partial_toml_keeps_other_defaults(tmp_path: Path) -> None:
    path = write_toml(tmp_path, "[ci_fix]\nmax_parallel_workers = 8\n")
    s = load_settings(path=path, env={})
    assert s.max_parallel_workers == 8
    assert s.model_dump(exclude={"max_parallel_workers"}) == Settings().model_dump(
        exclude={"max_parallel_workers"}
    )


def test_load_missing_ci_fix_table_gives_defaults(tmp_path: Path) -> None:
    path = write_toml(tmp_path, '[other]\nfoo = "bar"\n')
    assert load_settings(path=path, env={}) == Settings()


def test_load_empty_file_gives_defaults(tmp_path: Path) -> None:
    path = write_toml(tmp_path, "")
    assert load_settings(path=path, env={}) == Settings()


def test_load_invalid_toml_raises(tmp_path: Path) -> None:
    path = write_toml(tmp_path, "[ci_fix\nmodel = = 'x'\n")
    with pytest.raises(ConfigError):
        load_settings(path=path, env={})


# --------------------------------------------------------------------------- #
# load_settings: validation errors surface as ConfigError
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "body",
    [
        "temperature = 1.5",
        "temperature = -0.1",
        "max_attempts = 0",
        "max_parallel_workers = 0",
        "regression_timeout_seconds = 0",
        'regression_command = ["pytest"]',
        'max_attempts = "three"',
        "unknown_key = 1",
    ],
)
def test_load_invalid_values_raise_config_error(tmp_path: Path, body: str) -> None:
    path = write_toml(tmp_path, f"[ci_fix]\n{body}\n")
    with pytest.raises(ConfigError):
        load_settings(path=path, env={})


@pytest.mark.parametrize("key", ["anthropic_api_key", "github_token"])
def test_load_secrets_in_toml_rejected(tmp_path: Path, key: str) -> None:
    path = write_toml(tmp_path, f'[ci_fix]\n{key} = "leaked-secret"\n')
    with pytest.raises(ConfigError):
        load_settings(path=path, env={})


# --------------------------------------------------------------------------- #
# load_settings: secrets from env
# --------------------------------------------------------------------------- #


def test_load_reads_secrets_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    s = load_settings(
        env={"ANTHROPIC_API_KEY": FAKE_ANTHROPIC_KEY, "GITHUB_TOKEN": FAKE_GITHUB_TOKEN}
    )
    assert s.anthropic_api_key is not None
    assert s.github_token is not None
    assert s.anthropic_api_key.get_secret_value() == FAKE_ANTHROPIC_KEY
    assert s.github_token.get_secret_value() == FAKE_GITHUB_TOKEN
    s.require_secrets()


def test_load_empty_env_secrets_treated_as_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    s = load_settings(env={"ANTHROPIC_API_KEY": "", "GITHUB_TOKEN": ""})
    assert s.anthropic_api_key is None
    assert s.github_token is None


def test_load_uses_only_given_env_not_os_environ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-os-environ")
    monkeypatch.setenv("GITHUB_TOKEN", "from-os-environ")
    s = load_settings(env={})
    assert s.anthropic_api_key is None
    assert s.github_token is None


def test_load_defaults_env_to_os_environ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC_KEY)
    monkeypatch.setenv("GITHUB_TOKEN", FAKE_GITHUB_TOKEN)
    s = load_settings()
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == FAKE_ANTHROPIC_KEY


def test_loaded_settings_repr_hides_env_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    s = load_settings(
        env={"ANTHROPIC_API_KEY": FAKE_ANTHROPIC_KEY, "GITHUB_TOKEN": FAKE_GITHUB_TOKEN}
    )
    assert FAKE_ANTHROPIC_KEY not in repr(s)
    assert FAKE_GITHUB_TOKEN not in repr(s)


# --------------------------------------------------------------------------- #
# config.example.toml
# --------------------------------------------------------------------------- #


def test_example_config_exists() -> None:
    assert EXAMPLE_CONFIG.is_file()


def test_example_config_loads_and_matches_defaults() -> None:
    s = load_settings(path=EXAMPLE_CONFIG, env={})
    assert s == Settings()


def test_example_config_secrets_come_from_env() -> None:
    s = load_settings(
        path=EXAMPLE_CONFIG,
        env={"ANTHROPIC_API_KEY": FAKE_ANTHROPIC_KEY, "GITHUB_TOKEN": FAKE_GITHUB_TOKEN},
    )
    assert s.anthropic_api_key is not None
    assert s.anthropic_api_key.get_secret_value() == FAKE_ANTHROPIC_KEY
    assert s.model_dump(exclude={"anthropic_api_key", "github_token"}) == Settings().model_dump(
        exclude={"anthropic_api_key", "github_token"}
    )


@pytest.mark.parametrize("key", ["ANTHROPIC_API_KEY", "GITHUB_TOKEN", "Github_Token"])
def test_secret_in_toml_any_case_rejected_without_leaking(tmp_path: Path, key: str) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[ci_fix]\n{key} = "sk-super-secret-value"\n')
    with pytest.raises(ConfigError) as exc:
        load_settings(cfg, env={})
    assert "sk-super-secret-value" not in str(exc.value)
    assert "environment variables" in str(exc.value)


def test_unknown_key_error_does_not_echo_value(tmp_path: Path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text('[ci_fix]\napi_key = "sk-super-secret-value"\n')
    with pytest.raises(ConfigError) as exc:
        load_settings(cfg, env={})
    assert "sk-super-secret-value" not in str(exc.value)


@pytest.mark.parametrize("field", ["model", "branch_prefix"])
def test_empty_string_fields_rejected(tmp_path: Path, field: str) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[ci_fix]\n{field} = ""\n')
    with pytest.raises(ConfigError):
        load_settings(cfg, env={})


@pytest.mark.parametrize("value", ["default", "none", "DEFAULT", ""])
def test_temperature_default_means_not_sent(tmp_path: Path, value: str) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[ci_fix]\nmodel = "claude-sonnet-5-5"\ntemperature = "{value}"\n')
    s = load_settings(cfg, env={})
    assert s.temperature is None
    assert s.model == "claude-sonnet-5-5"
