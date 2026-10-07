# ci-fix

A Python library and CLI that takes a GitHub pull request with failing tests, fixes the
underlying bugs with an LLM agent, pushes the fix to a patch branch, and opens a PR with a
short, readable description.

**Status:** under construction (slice 0: project setup and config loader).

## Setup

```bash
uv sync
cp config.example.toml config.toml   # then edit as needed
export ANTHROPIC_API_KEY=...
export GITHUB_TOKEN=...
```

## Development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```
