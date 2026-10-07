# sample-calc (ci-fix test fixture)

A tiny Python project with seeded bugs. ci-fix tests copy it to a temp dir and run
pytest against it. No install is needed: `tests/conftest.py` puts `src` on `sys.path`.

The outer ci-fix test run does not collect these tests (`tests/fixtures/conftest.py`).

## Seeded bugs (`src/calc/ops.py`)

| Function | Bug | One-line fix |
|---|---|---|
| `subtract` | returns `a + b` | `return a - b` |
| `divide` | no zero check | raise `ValueError("division by zero")` when `b == 0` |
| `mean` | divides by `len(xs) + 1` | divide by `len(xs)` |

## Expected test results

| Node id | Status |
|---|---|
| `tests/test_ops.py::test_add` | passed |
| `tests/test_ops.py::test_subtract` | failed |
| `tests/test_ops.py::TestDivide::test_divide_ok` | passed |
| `tests/test_ops.py::TestDivide::test_divide_by_zero` | failed (raises `ZeroDivisionError`) |
| `tests/test_ops.py::test_mean[xs0-2]` | failed |
| `tests/test_ops.py::test_mean[xs1-15]` | failed |
| `tests/test_ops.py::test_skipped` | skipped |
| `tests/test_errors.py::test_fixture_error` | error (fixture raises `RuntimeError` at setup) |
| `tests/test_errors.py::test_add` | passed |

The bare name `test_add` is ambiguous (it exists in both test files).
