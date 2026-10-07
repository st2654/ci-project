"""Command-line entry point for ci-fix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ci_fix import __version__
from ci_fix.config import ConfigError, load_settings
from ci_fix.logging_setup import configure_logging, get_logger
from ci_fix.models import FixerFatalError, FixResult, OutcomeStatus
from ci_fix.pipeline import fix_failing_tests
from ci_fix.tools.git import GitError
from ci_fix.tools.github import GitHubError
from ci_fix.tools.pytest_runner import TestRunError
from ci_fix.tools.test_env import TestEnvError

EXIT_OK = 0
EXIT_NOT_ALL_FIXED = 1
EXIT_ERROR = 2
EXIT_INTERRUPTED = 130  # shell convention: 128 + SIGINT
_SUCCESS = (OutcomeStatus.FIXED, OutcomeStatus.ALREADY_PASSING)
_ERRORS = (ConfigError, GitError, GitHubError, TestEnvError, TestRunError, FixerFatalError)

log = get_logger(__name__)


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}") from None
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be positive: {value}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ci-fix", description="Fix failing tests on a GitHub pull request."
    )
    parser.add_argument("--version", action="version", version=f"ci-fix {__version__}")
    parser.add_argument("--repo", required=True, help="GitHub repository URL")
    parser.add_argument("--pr", required=True, type=_positive_int, help="pull request number")
    parser.add_argument(
        "--tests", required=True, nargs="+", metavar="ID", help="failing test ids or names"
    )
    parser.add_argument("--config", type=Path, help="path to config.toml")
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="console log level"
    )
    parser.add_argument("--log-file", type=Path, help="write a full DEBUG log to this file")
    parser.add_argument(
        "--keep-workspace", action="store_true", help="do not delete the workspace afterwards"
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="dry run: build the fix commit and PR description, but push/open/comment nothing",
    )
    parser.add_argument(
        "--no-comment", action="store_true", help="do not comment on the original PR"
    )
    return parser


def delivery_line(result: FixResult, push: bool) -> str:
    """One line saying where the fixes went (fix PR URL, dry run, no fixes, not pushed)."""
    if result.pr_url:
        return f"Fix PR: {result.pr_url}"
    if not result.fixed:
        return "No fixes: nothing was committed, pushed or opened."
    if not push:
        sha = f" {result.commit_sha[:12]}" if result.commit_sha else ""
        return f"Dry run: fix commit{sha} built locally; nothing pushed (see result.diff)."
    return "Fixes not pushed (see Warnings); the diff is in the result."


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        settings = load_settings(args.config)
    except ConfigError as exc:
        print(f"ci-fix: error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    overrides: dict[str, object] = {}
    if args.log_level is not None:
        overrides["log_level"] = args.log_level
    if args.log_file is not None:
        overrides["log_file"] = args.log_file.expanduser()
    if args.keep_workspace:
        overrides["keep_workspace"] = True
    if args.no_push:
        overrides["push"] = False
    if args.no_comment:
        overrides["comment_on_pr"] = False
    if overrides:
        settings = settings.model_copy(update=overrides)

    secrets = [
        s.get_secret_value()
        for s in (settings.github_token, settings.anthropic_api_key)
        if s is not None
    ]
    configure_logging(settings.log_level, settings.log_file, secrets=secrets)

    try:
        result = fix_failing_tests(args.repo, args.pr, args.tests, settings=settings)
    except _ERRORS as exc:
        print(f"ci-fix: error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("ci-fix: interrupted", file=sys.stderr)
        return EXIT_INTERRUPTED
    except Exception as exc:
        log.debug("Unexpected error", exc_info=True)
        first_line = next(iter(str(exc).splitlines()), "")
        print(f"ci-fix: unexpected error: {type(exc).__name__}: {first_line}", file=sys.stderr)
        return EXIT_ERROR

    print(delivery_line(result, settings.push))
    print()
    print(result.summary)
    if all(t.status in _SUCCESS for t in result.tests):
        return EXIT_OK
    return EXIT_NOT_ALL_FIXED


if __name__ == "__main__":
    raise SystemExit(main())
