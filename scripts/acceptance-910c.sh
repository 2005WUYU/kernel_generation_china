#!/bin/sh
set -eu

umask 077

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=${AKG_PROJECT_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)}
VENV_DIR=${AKG_VENV_DIR:-"${PROJECT_ROOT}-venv"}
CONFIG_PATH=${AKG_CONFIG_PATH:-"$PROJECT_ROOT/configs/experiment_910c_kimi_k3.yaml"}
CANN_ENV_FILE=${AKG_CANN_ENV_FILE:-}
ACCEPTANCE_OUTPUT=${AKG_ACCEPTANCE_OUTPUT:-"$PROJECT_ROOT/runs/acceptance_report_$(date -u +%Y%m%dT%H%M%SZ).json"}
EVIDENCE_ROOT=${AKG_ACCEPTANCE_EVIDENCE_ROOT:-"$PROJECT_ROOT/runs/acceptance_evidence"}
EXPERIMENT_ID=${AKG_EXPERIMENT_ID:-}

if [ "$#" -ne 0 ]; then
    echo "usage: configure AKG_* environment variables, then run $0 without arguments" >&2
    exit 2
fi
case "$PROJECT_ROOT:$VENV_DIR:$CONFIG_PATH" in
    /*:/*:/*) ;;
    *)
        echo "error: AKG_PROJECT_ROOT, AKG_VENV_DIR, and AKG_CONFIG_PATH must be absolute" >&2
        exit 2
        ;;
esac
if [ -n "$CANN_ENV_FILE" ]; then
    case "$CANN_ENV_FILE" in
        /*) ;;
        *)
            echo "error: AKG_CANN_ENV_FILE must be absolute" >&2
            exit 2
            ;;
    esac
fi

if [ -e "$ACCEPTANCE_OUTPUT" ]; then
    echo "error: acceptance output already exists; refusing to overwrite: $ACCEPTANCE_OUTPUT" >&2
    exit 2
fi

if [ -n "$CANN_ENV_FILE" ]; then
    if [ ! -r "$CANN_ENV_FILE" ]; then
        echo "error: CANN environment file is not readable: $CANN_ENV_FILE" >&2
        exit 2
    fi
    # shellcheck source=/dev/null
    . "$CANN_ENV_FILE"
fi

if [ ! -x "$VENV_DIR/bin/akg" ]; then
    echo "error: akg executable was not found in $VENV_DIR" >&2
    exit 2
fi
if [ ! -r "$CONFIG_PATH" ]; then
    echo "error: experiment config is not readable: $CONFIG_PATH" >&2
    exit 2
fi

cd "$PROJECT_ROOT"
# The command is read-only with respect to model/provider state: it checks the
# configured environment and verifies an already committed run. It does not
# call the model and does not execute candidates or fault-injection scenarios.
if [ -n "$EXPERIMENT_ID" ]; then
    exec "$VENV_DIR/bin/akg" acceptance \
        -c "$CONFIG_PATH" \
        --experiment-id "$EXPERIMENT_ID" \
        --evidence-root "$EVIDENCE_ROOT" \
        -o "$ACCEPTANCE_OUTPUT"
fi
exec "$VENV_DIR/bin/akg" acceptance \
    -c "$CONFIG_PATH" \
    --evidence-root "$EVIDENCE_ROOT" \
    -o "$ACCEPTANCE_OUTPUT"
