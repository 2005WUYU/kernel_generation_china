#!/bin/sh
set -eu

# Controller and Worker are separate UIDs in the same deployment group. Shared
# SQLite/WAL and committed artifacts therefore need group read/write access;
# candidate stage directories apply their own explicit 0700 mode.
umask 0007

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=${AKG_PROJECT_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)}
VENV_DIR=${AKG_VENV_DIR:-"${PROJECT_ROOT}-venv"}
CONFIG_PATH=${AKG_CONFIG_PATH:-"$PROJECT_ROOT/configs/experiment_910c_kimi_k3.yaml"}

case "$PROJECT_ROOT:$VENV_DIR:$CONFIG_PATH" in
    /*:/*:/*) ;;
    *)
        echo "error: AKG_PROJECT_ROOT, AKG_VENV_DIR, and AKG_CONFIG_PATH must be absolute" >&2
        exit 2
        ;;
esac
if [ ! -x "$VENV_DIR/bin/akg" ]; then
    echo "error: akg executable was not found in $VENV_DIR" >&2
    exit 2
fi
if [ ! -r "$CONFIG_PATH" ]; then
    echo "error: experiment config is not readable: $CONFIG_PATH" >&2
    exit 2
fi
"$VENV_DIR/bin/python" "$SCRIPT_DIR/validate-hidden-seed.py"

cd "$PROJECT_ROOT"
exec "$VENV_DIR/bin/akg" experiment resume -c "$CONFIG_PATH" "$@"
