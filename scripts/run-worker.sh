#!/bin/sh
set -eu

# The durable queue and control-plane artifact hierarchy are shared with the
# controller UID. Candidate attempts are still explicitly created as 0700.
umask 0007

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=${AKG_PROJECT_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)}
VENV_DIR=${AKG_VENV_DIR:-"${PROJECT_ROOT}-venv"}
CONFIG_PATH=${AKG_CONFIG_PATH:-"$PROJECT_ROOT/configs/experiment_910c_kimi_k3.yaml"}
CANN_ENV_FILE=${AKG_CANN_ENV_FILE:-}

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

reject_model_environment() {
    if [ "${ANTHROPIC_AUTH_TOKEN+x}" = x ] || \
       [ "${ANTHROPIC_API_KEY+x}" = x ] || \
       [ "${ANTHROPIC_BASE_URL+x}" = x ] || \
       [ "${ANTHROPIC_MODEL+x}" = x ] || \
       [ "${KIMI_API_KEY+x}" = x ] || \
       [ "${OPENAI_API_KEY+x}" = x ] || \
       [ "${AIPING_API_KEY+x}" = x ]; then
        echo "error: Worker inherited model/provider environment" >&2
        echo "use deploy/env/worker.env.example and never load controller.env" >&2
        exit 10
    fi
}

reject_model_environment

if [ -n "$CANN_ENV_FILE" ]; then
    if [ ! -r "$CANN_ENV_FILE" ]; then
        echo "error: CANN environment file is not readable: $CANN_ENV_FILE" >&2
        exit 2
    fi
    # shellcheck source=/dev/null
    . "$CANN_ENV_FILE"
fi

reject_model_environment

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
exec "$VENV_DIR/bin/akg" worker run -c "$CONFIG_PATH" "$@"
