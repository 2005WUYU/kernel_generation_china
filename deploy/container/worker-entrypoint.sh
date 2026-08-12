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

reject_model_environment() {
    if [ "${ANTHROPIC_AUTH_TOKEN+x}" = x ] || \
       [ "${ANTHROPIC_API_KEY+x}" = x ] || \
       [ "${ANTHROPIC_BASE_URL+x}" = x ] || \
       [ "${ANTHROPIC_MODEL+x}" = x ] || \
       [ "${KIMI_API_KEY+x}" = x ] || \
       [ "${OPENAI_API_KEY+x}" = x ] || \
       [ "${AIPING_API_KEY+x}" = x ]; then
        echo "error: Worker inherited model/provider environment" >&2
        exit 10
    fi
}

reject_model_environment
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
if [ -z "${AKG_HIDDEN_SEED:-}" ]; then
    echo "error: Worker hidden seed is missing" >&2
    exit 10
fi
if [ -z "${ASCEND_VISIBLE_DEVICES:-}" ]; then
    echo "error: ASCEND_VISIBLE_DEVICES must select the Worker NPU" >&2
    exit 3
fi
case "$ASCEND_VISIBLE_DEVICES" in
    *','*|*[!0-9]*)
        echo "error: Worker must expose exactly one numeric NPU" >&2
        exit 3
        ;;
esac

export AKG_PROJECT_ROOT="$PROJECT_ROOT"
export AKG_CONFIG_PATH="$CONFIG_PATH"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$PROJECT_ROOT"
python3 "$PROJECT_ROOT/scripts/validate-hidden-seed.py"

python3 - <<'PY'
import torch
import torch_npu  # noqa: F401
import triton  # noqa: F401

if not hasattr(torch, "npu") or not torch.npu.is_available():
    raise SystemExit("error: torch NPU is not available inside Worker container")
if torch.npu.device_count() != 1:
    raise SystemExit(
        f"error: Worker must see exactly one NPU, found {torch.npu.device_count()}"
    )
PY

reject_model_environment
if [ "$#" -gt 0 ]; then
    exec "$@"
fi
exec python3 -m ascend_kernel_lab worker run -c "$CONFIG_PATH"
