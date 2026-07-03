# Quality gate for manifold-sdk. Run `just check` before pushing.

[private]
default:
    @just --list

# The single gate: format check, lint, typecheck, test — all must be green.
check: fmt-check lint typecheck test

# Format the source and tests in place.
fmt:
    uv run ruff format src tests

# Fail if anything is unformatted (the gate uses this, not `fmt`).
fmt-check:
    uv run ruff format --check src tests

# Lint the source and tests.
lint:
    uv run ruff check src tests

# Type-check the SDK and its tests.
typecheck:
    uv run ty check src tests

# Run the test suite.
test:
    uv run pytest
