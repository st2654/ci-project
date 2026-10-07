"""Settings model and loader: non-secret values from TOML, secrets from the environment."""

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

# Relative to the CWD at startup. Load settings once at startup and pass the Settings
# object along; never reload after changing directory into a cloned workspace.
DEFAULT_CONFIG_PATH = Path("config.toml")
CONFIG_TABLE = "ci_fix"
SECRET_ENV_VARS: dict[str, str] = {
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "github_token": "GITHUB_TOKEN",
}


class ConfigError(Exception):
    """Raised when configuration is missing, malformed, or invalid."""


class Settings(BaseModel):
    """Runtime settings for ci-fix."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    model: str = Field(default="claude-sonnet-4-6", min_length=1)
    # None = don't send a temperature (use the model's default). Newer models such as
    # claude-sonnet-5-5 reject `temperature`; set it to "default" in config.toml for them.
    temperature: float | None = Field(default=0.0, ge=0.0, le=1.0)
    max_attempts: int = Field(default=3, ge=1)
    max_parallel_workers: int = Field(default=4, ge=1)
    workspace_dir: Path = Field(default=Path("~/.ci-fix/workspaces"), validate_default=True)
    # Must start with "ci-fix" (the branch is force-pushed; see delivery guards).
    branch_prefix: str = Field(default="ci-fix/pr-", min_length=1)
    pytest_args: list[str] = Field(default_factory=list)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: Path | None = None
    # Full-suite regression run (only after an attempt changed source files).
    regression_pytest_args: list[str] = Field(default_factory=list)
    regression_timeout_seconds: int = Field(default=1800, ge=1)
    python_version: str = Field(default="3.11", min_length=1)
    install_command: list[str] | None = None
    test_timeout_seconds: int = Field(default=900, ge=1)
    keep_workspace: bool = False
    max_agent_steps: int = Field(default=30, ge=1)
    max_output_tokens: int = Field(default=4096, ge=256)
    pr_diff_max_chars: int = Field(default=40000, ge=1000)
    read_max_lines: int = Field(default=400, ge=50)
    # Ask a second LLM call to approve test-file changes (after the patch checker passes).
    review_test_changes: bool = False
    # Delivery: False = dry run (squash commit + texts, but no push, PR or comment).
    push: bool = True
    # Comment on the original PR (link to the fix PR, or why nothing could be fixed).
    comment_on_pr: bool = True
    # Identity of the squashed fix commit (author and committer).
    commit_author_name: str = Field(default="ci-fix", min_length=1)
    commit_author_email: str = Field(default="ci-fix@users.noreply.github.com", min_length=3)
    anthropic_api_key: SecretStr | None = None
    github_token: SecretStr | None = None

    @field_validator("temperature", mode="before")
    @classmethod
    def _default_temperature(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in {"default", "none", ""}:
            return None
        return value

    @field_validator("branch_prefix", mode="after")
    @classmethod
    def _ci_fix_branch(cls, value: str) -> str:
        # The fix branch is force-pushed: it must be clearly ci-fix's own, never a user branch.
        if not value.startswith("ci-fix"):
            raise ValueError("branch_prefix must start with 'ci-fix'")
        return value

    @field_validator("workspace_dir", "log_file", mode="after")
    @classmethod
    def _expand_user(cls, value: Path | None) -> Path | None:
        return value.expanduser() if value is not None else None

    def require_secrets(self) -> None:
        """Raise ConfigError naming every required secret env var that is not set."""
        missing = [env for field, env in SECRET_ENV_VARS.items() if getattr(self, field) is None]
        if missing:
            raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")


def _read_toml(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc
    table = data.get(CONFIG_TABLE, {})
    if not isinstance(table, dict):
        raise ConfigError(f"[{CONFIG_TABLE}] in {path} must be a table")
    return table


def load_settings(path: Path | str | None = None, env: Mapping[str, str] | None = None) -> Settings:
    """Load settings from a TOML file (``[ci_fix]`` table) and secrets from ``env``.

    If ``path`` is None, ``./config.toml`` is used when it exists; otherwise defaults apply.
    """
    if env is None:
        env = os.environ

    values: dict[str, object] = {}
    if path is not None:
        config_path = Path(path)
        if not config_path.is_file():
            raise ConfigError(f"Config file not found: {config_path}")
        values = _read_toml(config_path)
    elif DEFAULT_CONFIG_PATH.is_file():
        values = _read_toml(DEFAULT_CONFIG_PATH)

    secret_names = set(SECRET_ENV_VARS) | {v.lower() for v in SECRET_ENV_VARS.values()}
    secret_keys = sorted(k for k in values if k.lower() in secret_names)
    if secret_keys:
        raise ConfigError(
            f"Config contains {', '.join(secret_keys)}: "
            "secrets must come from environment variables "
            f"({', '.join(SECRET_ENV_VARS.values())})"
        )

    for field, env_var in SECRET_ENV_VARS.items():
        values[field] = env.get(env_var) or None

    try:
        return Settings(**values)
    except ValidationError as exc:
        raise ConfigError(f"Invalid configuration: {exc}") from exc
