# CLAUDE.md — ci-fix

A Python library that takes a GitHub pull request with failing tests, fixes the
underlying bugs with an LLM agent, pushes the fix to a new patch branch, and
opens a PR with a concise, human-readable description.

---

## Working agreement (read first)

- **Always ask before assuming.** If a requirement, interface, or behaviour is
  unclear, stop and ask the user. Do not guess.
- **Never commit without review.** Before every `git commit`, stop, show the
  diff and proposed commit message, and wait for the user's explicit go-ahead.
- **Work in slices.** Build one slice at a time (see "Slices" below). A slice
  is done only when its acceptance criteria pass and the user has approved it.
- **Use helper agents per slice:**
  1. **Coder agent** writes the implementation.
  2. **Tester agent** writes tests for the same slice at the same time, working
     from the slice spec rather than the implementation.
  3. **Reviewer agent** reviews code and tests against this file before the
     user review.

---

## Decisions

| Topic | Decision |
|---|---|
| Language | Python 3.11+ |
| Package manager | `uv` |
| Orchestration | LangGraph (`StateGraph`) |
| LLM | Anthropic `claude-sonnet-5-5`, **temperature 0** (configurable) |
| Target test framework | **pytest only** (v1). Keep the runner behind an interface so others can be added later. |
| What a fix may change | Test files **and** source code. A source change must fix the real bug, never special-case the test (see "Integrity rules"). |
| Delivery | Push patch branch `ci-fix/pr-<N>` and **open a PR targeting the original PR's branch** (the fix layers on top of that PR) |
| Fork PRs | If the PR comes from a fork: push the fix to the fork's branch **only when** `maintainer_can_modify` is true and the token has access; otherwise skip the push/PR and return the fix as a diff with a clear message. |
| Execution | Run tests **locally** (Docker postponed). Each PR gets its own workspace `<workspace_dir>/<owner>__<repo>__pr-<N>/` with `repo/`, `venv/` (separate uv venv) and `reports/`; it is **deleted at the end of the run** unless `keep_workspace = true`. |
| Target-repo isolation | PR code runs with an **allowlisted environment**: its own venv, no `ANTHROPIC_API_KEY`/`GITHUB_TOKEN`/`PYTHON*`/`CI_FIX_*`. Timeouts kill the whole process group. |
| Test names | Full pytest node ids are used as given; bare names (`test_a`, `Cls::test_a`) are resolved via `pytest --collect-only`. Ambiguous or unknown names are reported, never guessed. |
| Fix attempts | **3** fix → re-test rounds per test, then report it as unfixable |
| Regression runs | Run the **full test suite** **only if the fix changed source code** (non-test files). Test-only fixes re-run just the target tests. |
| Secrets/config | Loaded from `config` + environment; the user fills in values (see "Configuration") |

---

## Public interface

```python
from ci_fix import fix_failing_tests

result = fix_failing_tests(
    repo_url="https://github.com/org/repo",
    pr_number=42,
    failing_tests=["tests/test_x.py::test_a", "tests/test_y.py::test_b"],
)
result.branch          # "ci-fix/pr-42"
result.pr_url          # URL of the opened fix PR
result.diff            # unified git diff of all fixes
result.summary         # markdown description (readable in 2–3 minutes)
result.tests           # per-test: FIXED | UNFIXABLE (+ reason, attempts)
```

CLI equivalent: `ci-fix --repo <url> --pr <n> --tests <id> [<id> ...]`

---

## Pipeline (LangGraph)

```
setup_repo ─► run_tests ─► triage ─► fix ─► verify ─┬─► regression? ─► finalize
                                       ▲            │
                                       └── failing & attempts < 3
```

1. **setup_repo** — Clone the repo, fetch `pull/<N>/head`, create branch
   `ci-fix/pr-<N>` from it.
2. **run_tests** — Run only the given tests with pytest, write results as JUnit
   XML, and parse that file (never scrape the console). Record the failure
   message and traceback for each test. If a test the user named already passes,
   report it as such and don't touch it.
3. **triage** — Decide **parallel vs sequential**:
   - Parallel only if the failures are independent (no overlapping files in
     their tracebacks) and the suite tolerates isolated runs.
   - Otherwise run fixes one after another.
4. **fix** — One Claude agent per failing test (temperature 0) with tools:
   `read_file`, `search_code`, `edit_file`, `run_test`. Parallel fixes run in
   separate git worktrees (LangGraph `Send` fan-out); patches are merged
   afterwards. If two patches conflict, redo those fixes sequentially.
5. **verify** — Re-run the target tests. Failures loop back to **fix** until
   the 3-attempt limit is reached.
6. **regression** — Only if any non-test file changed: run the regression
   suite. A new failure counts as a failed fix attempt.
7. **finalize** — Commit with a descriptive message, push the branch, open
   the PR with the summary, and return `FixResult`.

---

## Integrity rules (very important)

The goal is to fix the **real bug**, never to make a test pass by hiding it.

**Forbidden in test files:**
- Adding `skip`, `skipif`, `xfail`, or similar markers, or deleting a test.
- Removing or weakening assertions (e.g. `assert x == 5` → `assert x`),
  widening tolerances, or changing expected values just to match wrong output.
- Wrapping the failing code in `try/except` that swallows the error.

**Forbidden in source code:**
- Special-casing tests: checking for pytest or test env vars, hard-coding the
  expected test values, or branching on test inputs.
- Silencing errors (bare `except`, returning defaults on exception) to avoid
  the failure.

**Changing source code — be conservative and critical:**
- The developer wrote the source with context you may not have (product
  requirements, callers elsewhere, intended behaviour). Treat the existing
  source as intentional until the evidence says otherwise.
- Before editing source, the agent must state: (1) why the test is correct
  and the source is wrong, (2) what evidence supports that (PR diff, docstrings,
  other callers, other passing tests), and (3) what else could be affected.
- Prefer the smallest change that fixes the root cause. No refactors, renames,
  or unrelated clean-ups.
- If it is unclear whether the test or the source holds the intended
  behaviour, do **not** guess: mark the test UNFIXABLE with the reason
  "intent ambiguous" and explain both options.
- Every source change is called out in the PR description with this reasoning.

**Allowed:**
- Changing a test's expected value only when the test is demonstrably wrong.
  The agent must explain why, and the summary must flag the change.

**Enforcement:** a patch checker runs on every proposed diff and rejects
any of the forbidden patterns above. A rejected patch counts as a failed
attempt.

**Unfixable tests:** if a test cannot be fixed honestly (e.g. it needs an
external service or credentials, the requirement is ambiguous, or attempts
ran out), mark it **UNFIXABLE** with a clear reason and leave it failing. Never
disable or work around it.

---

## Commit message & PR description

- Concise; a person should understand it in **2–3 minutes**.
- Structure:
  - **Summary** — one or two sentences.
  - **Fixed** — per test: root cause (one line) → what changed (one line), files.
  - **Test files changed** (if any) — with justification.
  - **Unfixable** — per test: the reason.
  - **Verification** — what was re-run and the result.
- Add short code comments at each fix site only where the *why* is not obvious.

---

## Tools to build

- **GitHub** (`PyGithub` or REST via `httpx`): get PR metadata, push the
  branch, open a PR, comment on the original PR.
- **Git** (subprocess wrapper): clone, fetch PR ref, branch, worktree add or
  remove, diff, commit, push.
- **Test runner**: run pytest node IDs, parse JUnit XML into results.
- **Agent tools**: `read_file`, `search_code`, `edit_file` (restricted to the
  repo or worktree root), `run_test`.

---

## Configuration

`config.example.toml` is committed. Copy it to `config.toml` (gitignored) and
fill in the values. Secrets come from the environment:

- `ANTHROPIC_API_KEY`
- `GITHUB_TOKEN` (needs repo + pull request write access)

Configurable: model name, temperature, max attempts, parallel worker count,
workspace directory, pytest extra arguments, regression test command.

---

## Project layout (proposed)

```
src/ci_fix/
  __init__.py        # fix_failing_tests()
  cli.py
  config.py
  models.py          # FixResult, TestOutcome, PipelineState
  graph.py           # LangGraph wiring
  nodes/             # setup_repo, run_tests, triage, fix, verify, regression, finalize
  tools/             # git.py, github.py, pytest_runner.py, agent_tools.py
  guards/patch_checker.py
  prompts/
tests/
  fixtures/sample_repo/   # small repo with seeded bugs for end-to-end tests
```

---

## Slices

| # | Slice | Acceptance |
|---|---|---|
| 0 | Project setup: `uv`, ruff, pytest, config loader, `config.example.toml` | `uv run pytest` is green |
| 1 | Git + GitHub tools: clone PR, create branch | Works against a real public PR |
| 2 | Pytest runner + JUnit XML parsing | Correct results on the sample repo |
| 3 | LangGraph skeleton with a stub fixer: state, retry loop, 3-attempt limit | Loop and exit paths tested |
| 4 | Claude fixer agent (sequential), temperature 0 | Fixes seeded bugs in the sample repo |
| 5 | Patch checker + UNFIXABLE reporting | Rejects every forbidden pattern |
| 6 | Conditional regression run (only on source changes) | Runs only when source changed |
| 7 | Parallel fixing (worktrees + `Send`) | Same results as sequential |
| 8 | Finalize: commit, summary, push, open PR | PR description meets the 2–3 min bar |
| 9 | End-to-end run on the sample repo | Full pipeline passes |
| — | *Future:* Docker sandbox, other test frameworks | — |

---

## Logging

- Use `log = get_logger(__name__)` from `ci_fix.logging_setup`; never `print` in library code.
- The library never configures handlers on import (a `NullHandler` is installed). The CLI or
  caller calls `configure_logging(level, log_file, secrets)` once at startup.
- **INFO** = progress a user wants to watch. Pipeline steps are prefixed like
  `[setup 2/4] Cloning ...`, with durations for slow steps.
- **DEBUG** = troubleshooting detail: every git and pytest command, its duration, and its stderr.
  Lower-level tools (git, GitHub) log at DEBUG; pipeline steps own the INFO lines.
- **WARNING/ERROR** = something went wrong or needs attention (e.g. a test marked UNFIXABLE).
- Secrets must never be logged. Mask them with `_redact` / `RedactSecretsFilter`, and test that.
- Config: `log_level` (console, default INFO) and optional `log_file` (full DEBUG log).

## Coding conventions

- Type hints everywhere; Pydantic models for state and results.
- No network calls in unit tests; mock GitHub and the LLM. The end-to-end
  slice uses the real APIs only when credentials are set.
- Keep nodes pure where possible: take state in, return a state update.
