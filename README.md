# ci-fix

Give it a GitHub pull request and the names of its failing tests. ci-fix finds the root cause
of each failure with a Claude agent, fixes it, proves the fix with the test suite, and opens a
**fix PR against your PR's branch** with a description a reviewer can read in 2–3 minutes.

If a test cannot be fixed honestly, ci-fix says so and why. It never skips, deletes or weakens
a test to make it pass.

```
$ ci-fix --repo https://github.com/org/repo --pr 42 --tests test_area test_slugify test_fetch_status
...
Fix PR: https://github.com/org/repo/pull/43
```

## How it works

```
clone PR ─► test env ─► resolve names ─► run tests ─► fix (parallel per group) ─► verify ─► deliver
                                                         ▲                          │
                                                         └── rejected: roll back, ──┘
                                                             retry (≤ 3 attempts)
```

1. **Setup** — clones the repo, checks out the PR head on `ci-fix/pr-<N>`, creates an isolated
   Python environment for the PR (`uv`), and resolves test names (`test_a`, `Cls::test_a` or full
   pytest ids).
2. **Fix** — a Claude agent (default `claude-sonnet-4-6`, temperature 0) reads the failure, the
   code and the PR diff, and makes the smallest fix. Independent failures are fixed in parallel
   in separate git worktrees.
3. **Verify** — every attempt must pass:
   - the **patch checker** (no skipped/deleted/weakened tests, no dummy fixtures, no
     monkeypatching the code under test, no special-casing test inputs, …);
   - **all requested tests** re-run;
   - the **full test suite** when source code or shared test code changed (regressions are
     rejected);
   - optionally a second **Claude reviewer** for any test-file change.
   Rejected attempts are rolled back and the reason is given to the next attempt.
4. **Deliver** — accepted fixes are squashed into one commit, pushed to `ci-fix/pr-<N>`, and a
   PR is opened (or updated on re-runs) against the original PR's branch, plus a comment on the
   original PR.

The fix PR lists, per test: root cause, fix and files; **test expectation changes** (flagged,
with the justification); **source changes** (flagged for review, since the original author may
have context the agent lacked); unfixable tests with reasons; pre-existing failures it did not
touch; and what was verified.

## Install

Requires Python 3.11+, git and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/st2654/ci-project && cd ci-project
uv sync
cp config.example.toml config.toml   # optional; defaults are fine
export ANTHROPIC_API_KEY=...
export GITHUB_TOKEN=...
```

**Token scopes.** Use a fine-grained GitHub token limited to the repos you'll fix, with
*Contents: read & write* and *Pull requests: read & write*. Secrets are only read from the
environment, never from `config.toml`.

## Usage

```bash
uv run ci-fix --repo URL --pr N --tests ID [ID ...] \
  [--config PATH] [--log-level DEBUG|INFO|WARNING|ERROR] [--log-file PATH] \
  [--keep-workspace] [--no-push] [--no-comment]
```

- `--no-push` — dry run: fixes, commit message and PR body are produced, nothing is sent to GitHub.
- `--keep-workspace` — keep the clone for inspection (deleted by default).
- `--log-file` — full DEBUG log for troubleshooting (console shows progress at INFO).

Exit codes: `0` everything fixed or already passing · `1` some tests not fixed ·
`2` configuration/GitHub/git/API error · `130` interrupted.

From Python:

```python
from ci_fix import fix_failing_tests

result = fix_failing_tests("https://github.com/org/repo", 42, ["test_area", "test_slugify"])
result.pr_url, result.diff, result.summary
for t in result.tests:
    print(t.requested_name, t.status, t.reason)   # fixed / already_passing / unfixable / not_found / ambiguous
```

## Configuration (`config.toml`, `[ci_fix]` table)

| Setting | Default | |
|---|---|---|
| `model` / `temperature` | `claude-sonnet-4-6` / `0.0` | Newer models reject temperature: set `temperature = "default"` |
| `max_attempts` | `3` | fix → verify rounds per test |
| `max_parallel_workers` | `4` | `1` = strictly sequential |
| `review_test_changes` | `false` | second Claude opinion on test-file changes |
| `regression_pytest_args` / `regression_timeout_seconds` | `[]` / `1800` | full-suite run options |
| `push` / `comment_on_pr` | `true` / `true` | delivery switches |
| `keep_workspace`, `workspace_dir`, `log_level`, `log_file`, … | | see `config.example.toml` |

## Safety model

- The PR's code runs in its own virtualenv with an allow-listed environment — no API keys or
  GitHub token are visible to it; timeouts kill the whole process tree.
- git runs with hooks and fsmonitor disabled, so a PR cannot plant code that runs with secrets.
- ci-fix only force-pushes branches it created and only edits PRs/comments it created.
- LLM text is sanitised before it reaches GitHub (no @mentions, no auto-closing keywords).

## Limitations

- Tests run locally on the machine running ci-fix (no Docker sandbox yet) — only run it on PRs
  you would run locally anyway.
- pytest only (unittest-style tests work through pytest).
- Parallel worktrees put the worktree's code first on `PYTHONPATH`; unusual package layouts
  may fall back to sequential behaviour.
- Fork PRs are pushed only when the author allowed maintainer edits; otherwise the diff is
  returned.

## Development

```bash
uv run pytest                     # ~950 offline tests
CI_FIX_RUN_INTEGRATION=1 uv run pytest -m integration   # real GitHub (+ Claude if ANTHROPIC_API_KEY is set)
uv run ruff check . && uv run ruff format --check .
```

Design notes and the working agreement for contributors are in [CLAUDE.md](CLAUDE.md).
