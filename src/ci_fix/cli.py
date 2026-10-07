"""Command-line entry point for ci-fix."""

import argparse
import sys

from ci_fix import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ci-fix", description="Fix failing tests on a GitHub pull request."
    )
    parser.add_argument("--version", action="version", version=f"ci-fix {__version__}")
    parser.parse_args(argv)
    print("ci-fix: not implemented yet.", file=sys.stderr)
    return 2  # non-zero until the pipeline exists, so CI never mistakes this for success


if __name__ == "__main__":
    raise SystemExit(main())
