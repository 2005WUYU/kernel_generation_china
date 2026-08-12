#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    PYTHON=$PROJECT_ROOT/.venv/bin/python
else
    PYTHON=${AKG_DEV_PYTHON:-python3}
fi

cd "$PROJECT_ROOT"

echo "[1/7] Ruff"
"$PYTHON" -m ruff check src tests

echo "[2/7] mypy"
"$PYTHON" -m mypy src

echo "[3/7] CPU tests"
"$PYTHON" -m pytest -m 'not npu'

echo "[4/7] CLI"
"$PYTHON" -m ascend_kernel_lab --help >/dev/null

echo "[5/7] shell syntax"
for script in scripts/*.sh; do
    sh -n "$script"
done

echo "[6/7] shellcheck"
if command -v shellcheck >/dev/null 2>&1; then
    shellcheck scripts/*.sh
else
    echo "shellcheck is not installed; sh -n completed"
fi

echo "[7/7] Git whitespace"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git diff --check
else
    echo "not a Git worktree; skipped git diff --check"
fi

echo "Local checks passed"
