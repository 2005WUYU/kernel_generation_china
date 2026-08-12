#!/bin/sh
set -eu

umask 0007

HOME=${HOME:-/tmp/akg-home}
case "$HOME" in
    /tmp/*) ;;
    *) echo "error: container HOME must be below /tmp" >&2; exit 2 ;;
esac
mkdir -p "$HOME"
chmod 0700 "$HOME"
export HOME

PROJECT_ROOT=${AKG_PROJECT_ROOT:-/workspace}
CONFIG_PATH=${AKG_CONFIG_PATH:-$PROJECT_ROOT/configs/experiment_910c_kimi_k3.yaml}
export GIT_OPTIONAL_LOCKS=0

case "$PROJECT_ROOT:$CONFIG_PATH" in
    /*:/*) ;;
    *)
        echo "error: AKG_PROJECT_ROOT and AKG_CONFIG_PATH must be absolute" >&2
        exit 2
        ;;
esac

if [ ! -r "$CONFIG_PATH" ] || [ ! -d "$PROJECT_ROOT/.git" ]; then
    echo "error: mount a complete clean Git clone at $PROJECT_ROOT" >&2
    exit 2
fi
TOPLEVEL=$(git -C "$PROJECT_ROOT" rev-parse --show-toplevel 2>/dev/null || true)
HEAD=$(git -C "$PROJECT_ROOT" rev-parse --verify HEAD 2>/dev/null || true)
if [ "$TOPLEVEL" != "$PROJECT_ROOT" ]; then
    echo "error: mounted project path is not the Git top-level" >&2
    exit 3
fi
case "$HEAD" in
    ????????????????????????????????????????|????????????????????????????????????????????????????????????????) ;;
    *) echo "error: mounted project has no full Git commit" >&2; exit 3 ;;
esac
case "$HEAD" in
    *[!0-9a-f]*) echo "error: Git commit must be lowercase hexadecimal" >&2; exit 3 ;;
esac
if [ -n "$(git -C "$PROJECT_ROOT" status --porcelain=v1 --untracked-files=all)" ]; then
    echo "error: mounted project clone is not clean" >&2
    exit 3
fi
if [ -z "${ANTHROPIC_BASE_URL:-}" ] || [ -z "${ANTHROPIC_AUTH_TOKEN:-}" ] || \
   [ -z "${ANTHROPIC_MODEL:-}" ]; then
    echo "error: controller model endpoint/token/model is missing" >&2
    exit 10
fi
if [ -z "${AKG_HIDDEN_SEED:-}" ]; then
    echo "error: controller hidden seed is missing" >&2
    exit 10
fi

export AKG_PROJECT_ROOT="$PROJECT_ROOT"
export AKG_CONFIG_PATH="$CONFIG_PATH"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
python3 "$PROJECT_ROOT/scripts/validate-hidden-seed.py"

if [ "$#" -gt 0 ]; then
    exec "$@"
fi
exec python3 -m ascend_kernel_lab experiment resume -c "$CONFIG_PATH"
