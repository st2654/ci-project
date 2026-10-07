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
| LLM | Anthropic `claude-sonnet-4-6`, **temperature 0** (both configurable). Newer models (`claude-sonnet-5-5`, `claude-opus-5-5`) reject `temperature`; for them set `temperature = "default"` so it isn't sent. Fatal API errors (400/401/403/404: bad key, no credit, unsupported parameter) stop the run instead of using up attempts. |
| Target test framework | **pytest only** (v1). Keep the runner behind an interface so others can be added later. |
| What a fix may change | Test files **and** source code. A source change must fix the real bug, never special-case the test (see "Integrity rules"). |
| Delivery | Push patch branch `ci-fix/pr-<N>` and **open a PR targeting the original PR's branch** (the fix layers on top of that PR) |
| Fork PRs | If the PR comes from a fork: push the fix to the fork's branch **only when** `maintainer_can_modify` is true and the token has access; otherwise skip the push/PR and return the fix as a diff with a clear message. |
| Execution | Run tests **locally** (Docker postponed). Each run gets its own workspace `<workspace_dir>/<owner>__<repo>__pr-<N>__<YYYYmmdd-HHMMSS>-<6 hex>/` (unique, so concurrent runs on the same PR never collide) with `repo/`, `venv/` (separate uv venv) and `reports/`; it is **deleted at the end of the run** unless `keep_workspace = true`. |
| Target-repo isolation | PR code runs with an **allowlisted environment**: its own venv, no `ANTHROPIC_API_KEY`/`GITHUB_TOKEN`/`PYTHON*`/`CI_FIX_*`. Timeouts kill the whole process group. PR tests can write to `.git`, so every git call runs with hooks and fsmonitor disabled (`-c core.hooksPath=/dev/null -c core.fsmonitor=false`) and without those secret env vars. |
| Test names | Full pytest node ids are used as given; bare names (`test_a`, `Cls::test_a`) are resolved via `pytest --collect-only`. Ambiguous or unknown names are reported, never guessed. |
| Fix attempts | **3** fix → re-test rounds per test, then report it as unfixable |
| Regression runs | Run the **full test suite** **only if the fix changed source code** (non-test files, per the patch checker) **or shared test code** (`conftest.py`, test-dir modules without tests such as helpers/factories, test data files). Baseline (fix stashed) at the first source change, then one full run per source-changing attempt; a test that passed in the baseline and now fails (after one flaky re-run) rejects the attempt. Test-only fixes re-run just the requested tests. Config: `regression_pytest_args`, `regression_timeout_seconds` (default 1800). |
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
setup_repo ─► setup_env ─► resolve_tests ─► run_initial ─► select_next ─┬─► finalize
                                                               ▲        │ (pending empty)
                                                               │        ▼
                                                          verify_one ◄─ fix_one
                                            (fix_one skips verify when the attempt is
                                             rejected early or the fixer gives up)

select_next ─► plan_round ═► fix_in_worktree (×K, Send) ─► merge_candidates ─► select_next
              (parallel mode: ≥2 independent groups pending and max_parallel_workers > 1)
```

1. **setup_repo** — Clone the repo, fetch `pull/<N>/head`, create branch
   `ci-fix/pr-<N>` from it. A failure part-way removes the half-made workspace.
2. **setup_env / resolve_tests** — Create the test venv; resolve bare names to node ids.
3. **run_initial** — Run all requested tests with pytest, write results as JUnit
   XML, and parse that file (never scrape the console). Tests that already pass
   are reported as such and not touched; failing ones become `pending`.
4. **Fix one test at a time** (in the order the user gave them) — *checkpoint &
   rollback*. With `max_parallel_workers > 1` only the fixer (agent) work may run in
   parallel; see "Parallel fixing" below.
   - **fix_one** calls the fixer for the next pending test. What changed is read
     from git, not from the fixer's claim. No change, or a fixer error, is a
     rejected attempt; a fixer that reports *unfixable* has its edits rolled back.
   - **verify_one** re-runs **all** requested tests, not just the target. The
     attempt is accepted only if the target now passes and nothing that passed
     before broke, disappeared (deleted/renamed) or became skipped.
   - **Flaky tests:** if the only problem is previously passing tests now failing,
     those are re-run once and the attempt is rejected only if they still fail.
     A test that seems fixed as a side effect is re-run once before it counts.
     *Limitation:* one re-run only catches occasional flakiness; a test that fails
     often, or the target itself passing by luck, can still give a wrong verdict.
   - **Artifacts:** untracked files left by env setup, the initial run and each
     verify run (`*.egg-info`, `.coverage`, …) are added to `.git/info/exclude`,
     so they are never committed, never in the diff and never deleted by rollback.
     The fixer's own new files are captured right after it returns.
   - **Accepted** attempts become a local **checkpoint commit**
     (`ci-fix: fix <id> (attempt <n>)`). Other pending tests that now pass are
     marked FIXED ("fixed by the fix for <id>") without calling the fixer.
   - **Rejected** attempts are **rolled back** (`reset --hard` + `clean -fd`), and
     the rejection reason is passed to the next attempt. After `max_attempts`
     (default 3) the test is UNFIXABLE with the last reason.
   - **Parallel fixing** (slice 7, `ci_fix/triage.py`): at each `select_next` the
     pending tests are grouped. Tests are in one group when the repo files mentioned in
     their failure tracebacks (`path.py:12:` and `File "path.py", line N`; files outside
     the repo, site-packages and venvs ignored) overlap, transitively, or when they are in
     the same test file. `conftest.py` frames are ignored for linking (shared fixtures would
     glue unrelated tests together); grouping that is too fine only costs a sequential retry
     when patches conflict. If `max_parallel_workers == 1` or there is only one group, the
     sequential path above runs unchanged. Otherwise a round (`plan_round`) takes the
     first pending test of up to `max_parallel_workers` groups and fans out with LangGraph
     `Send` to `fix_in_worktree`: each runs the fixer in its own detached git worktree
     (`<run_dir>/worktrees/r<round>-<n>`, at the current HEAD, removed afterwards) with its
     own runner (reports under `reports/worktrees/`), then captures the fixer's change set
     as a raw-bytes binary diff (`GitRepo.patch`, never decoded, so CRLF, non-UTF-8,
     binary files and mode changes round-trip exactly). Files a branch's own test runs create
     are kept in a private per-branch set and left out of its patch (`patch(skip=...)`);
     branches never write the `info/exclude` file, which all worktrees share and which could
     otherwise hide another branch's new file. Writes to `info/exclude` (main checkout) are
     deduplicated under a lock. Fixer errors become rejected attempts; `FixerFatalError`
     stops the run — LangGraph first waits for the branches still running (each removes
     its own worktree), then the error propagates. **Acceptance stays serial:** `merge_candidates` takes the candidates in `pending`
     order, applies each patch to the main checkout (`git apply`) and runs the same
     pipeline as the sequential path (`after_fixer`: changed files, unfixable, integrity
     check; `judge_and_commit`: verify, reviewer, regression, checkpoint/rollback). A
     candidate whose test was already fixed by an earlier candidate is dropped. A patch that
     no longer applies (it conflicts with a fix accepted earlier in the round) uses no
     attempt and is retried sequentially before the next round (`[parallel] patch for X
     conflicts with an accepted fix; retrying sequentially`).
   - *Parallel limitation:* the test venv's editable install points at the main checkout,
     so a worktree runner sets `PYTHONPATH` to `<worktree>/src` (if it exists) and the
     worktree root to make the worktree's code win. Other layouts (other package dirs,
     `package_dir` mappings, compiled extensions) may still import the main checkout's code
     while the fixer runs its test; the serial verification in the main checkout is
     unaffected. Tests can plug in their own runner via `PipelineDeps.worktree_runner_factory`
     (default: `runner_factory` when that is not the default one).
5. **regression** (slice 6, inside verify_one) — Only for an attempt that would
   otherwise be accepted and changed a source (non-test) file:
   - **Baseline** (once per run): the attempt is stashed (`git stash -u`), the full
     suite runs at HEAD (`PytestRunner.run_all`, pytest's own config decides what is
     collected, plus `regression_pytest_args`, timeout `regression_timeout_seconds`),
     then the attempt is restored (`stash pop --index`, always, even if the run fails).
   - The full suite runs again with the attempt applied. **Regressions** = tests that
     PASSED in the baseline and now fail, error or are missing (requested tests are
     judged by verify as before). They are re-run once; those that pass are treated as
     flaky. Any left → rollback and reject (`broke other tests in the full suite: a, b,
     … and N more`), a failed attempt whose reason goes to the next one. If none, the
     attempt is accepted and its full-suite results become the new baseline.
   - The reason lists ids now failing, then `; no longer collected: ...` (up to 10 each).
     The flaky re-run uses the same `regression_pytest_args`/timeout.
   - If a full run (baseline or with the fix) cannot complete (e.g. timeout), a WARNING
     is logged, regression checks are skipped for the rest of the run, the attempt is
     judged on the normal rules (no attempt is burnt on a slow suite) and
     `FixResult.warnings` gets `regression check skipped: ...`.
   - **Restoring the attempt after the baseline:** before the pop, tracked files are
     reset to HEAD and files the run created at the attempt's new paths are deleted
     (the run's other new files are excluded as artifacts). If the pop still fails, or
     the restored changed-file list differs from the one before the stash, the tree is
     reset to HEAD, the stash dropped and the attempt rejected (`could not restore the
     attempt after the baseline run: ...`, counts as an attempt). Checkpoint commits are
     never touched.
   - *Accepted limitations:* requested tests run twice for a source change (verify +
     full run), for simplicity. Tests in a file that already failed to import in the
     baseline have no individual baseline status, so they are not guarded one by one.
   - **Pre-existing failures:** tests outside the requested ones that fail in the
     latest baseline are returned in `FixResult.preexisting_failures` (slice 8 lists
     them in the PR description). Full-run artifacts are excluded like verify's.
6. **finalize** — The diff against the PR head contains **only accepted fixes**
   (the checkpoint commits). Slice 8 squashes the checkpoints into one commit with
   a descriptive message, pushes the branch, opens the PR and returns `FixResult`.

Per-run handles (workspace, test runner) live in a `RunContext` passed through the
LangGraph config; `PipelineDeps` is immutable, so one `deps` can serve concurrent runs.

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

**Enforcement:**
- **Patch checker** (`ci_fix/guards/patch_checker.py`, stdlib `ast`, always on) runs
  right after the fixer returns, comparing every changed `.py` file and pytest config
  file between `HEAD` and the working tree. Any violation rolls the attempt back and
  rejects it (`integrity check failed:` + the list of violations, fed back to the agent);
  it counts as a failed attempt; a checker crash is also a rejection
  (`integrity check failed: checker error: ...`). Test files are test modules
  (`python_files` from the pytest config at HEAD, default `test_*.py`/`*_test.py`,
  plus `conftest.py`) and every `.py` file in a `tests`/`test`/`testing` directory or a
  configured `testpaths` entry; they never get source rules. Rule ids (also used in the
  PR description):
  - any `.py`: `syntax_error`, `unparseable`, `unreadable`.
  - test files: `test_removed` (test or test class deleted/renamed), `skip_added`,
    `assertion_removed` (fewer asserts / `assert*` calls / `pytest.raises`),
    `trivial_assertion`, `test_emptied`, `expects_exception_added` (new
    `pytest.raises`/`assertRaises*` in a test that had none), `try_added`,
    `fixture_removed` (test parameter or `@pytest.fixture` removed), `param_removed`
    (fewer literal parametrize cases), `fixture_stubbed` (an existing fixture or
    `setUp`/`setup_method`-style method now only returns/yields/assigns a dummy —
    constant, `None`, `object()`, `Mock()`, empty literal, lambda — or its `raise` was
    replaced by returning one), `code_under_test_patched` (new `monkeypatch.setattr/
    setitem`, `mock.patch`/`patch.object`, `setattr`, rebinding imported names — new
    fixtures included — or importing a name from `mock`/`unittest.mock` or from a `.py`
    file created by the same patch), `collection_tampering`
    (`collect_ignore*`/`pytest_plugins` changed, or pytest hooks added/changed: collection,
    `pytest_runtest_*`, `makereport`, `report_teststatus`, `sessionfinish`, `configure`,
    `pyfunc_call` — also in modules registered via `pytest_plugins`),
    `test_data_changed` (binary file in a test directory changed),
    `unjustified_test_change`.
  - source files: `test_detection` (new pytest imports; "pytest"/"CI_FIX" strings used in
    a comparison, `in` test, subscript or lookup call — not in log messages or f-strings;
    `PYTEST_CURRENT_TEST`; `TESTING` env lookups), `special_case_inputs` (new
    `==`/`!=`/`in`/`is` comparison of a name with a literal from the failing test's file;
    `None`/`True`/`False`/`0`/`1`/`-1`/`""` ignored), `hardcoded_return` (an existing
    function now only returns a literal), `error_swallowed` (new bare/`Exception` handler
    that passes, continues or returns a constant).
  - config: `test_config_changed` (`pytest.ini`/`.pytest.ini`, or the pytest section of
    `tox.ini`, `setup.cfg`, `pyproject.toml`).
- **`Test change:` requirement:** a changed expectation (an assertion, a tolerance such
  as `abs=`/`rel=`/`places=`/`delta=`, any other change to an existing fixture or
  setup/teardown method, a changed or removed import of non-stdlib code in a test file
  (test `<path>::imports`, e.g. after the PR renamed a function), a changed text file in a
  test directory (data, snapshots), the
  literal value assigned to a variable an
  assertion uses, or a module-level literal in a test file) is allowed only if the
  fixer's explanation contains a line `Test change: <why the old expectation was wrong>`;
  otherwise `unjustified_test_change`. Accepted changes are recorded per test
  (`TestOutcome.test_changes`, `source_changed`, `explanation`) for the PR description.
- **Optional reviewer** (`review_test_changes = true`, default off): after an attempt
  that changed a test file passes the checker and verification, one extra Claude call
  (`ClaudeReviewer`) judges whether every test change is legitimate and rejects when in
  doubt (`reviewer rejected test change: <reason>`, also a failed attempt). It sees the
  test-file hunks first and in full; only source hunks are truncated. A reviewer error is
  retried once; if it fails twice the attempt is rejected (`reviewer unavailable: ...`,
  counts as an attempt). Fatal API errors stop the run.

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
workspace directory, pytest extra arguments, full-suite (regression) pytest arguments
and timeout.

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
  guards/              # patch_checker.py (integrity rules), reviewer.py (optional LLM review)
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
| 6 | Conditional regression run (only on source changes): baseline, full run per attempt, flaky re-run, pre-existing failures | Runs only when source changed; a new full-suite failure rejects the attempt |
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
