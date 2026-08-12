#!/bin/sh
set -eu

# The installed venv is shared read-only by two service UIDs. Build it with
# conventional traversal/read modes, then remove every group/other write bit.
umask 0022

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
VENV_DIR=${AKG_VENV_DIR:-"${PROJECT_ROOT}-venv"}
CANN_ENV_FILE=${AKG_CANN_ENV_FILE:-}
SYSTEM_PYTHON=${AKG_SYSTEM_PYTHON:-python3}
SHARED_GROUP=${AKG_SHARED_GROUP:-ascend-kernel}
VENV_OWNER=${AKG_VENV_OWNER:-root}
ALLOW_NON_GIT_SOURCE=${AKG_ALLOW_NON_GIT_SOURCE_FOR_TESTING:-0}

if [ "$#" -ne 0 ]; then
    echo "usage: configure AKG_* environment variables, then run $0 without arguments" >&2
    exit 2
fi

if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
    echo "error: pyproject.toml was not found under $PROJECT_ROOT" >&2
    exit 2
fi
if [ "$ALLOW_NON_GIT_SOURCE" != 0 ] && [ "$ALLOW_NON_GIT_SOURCE" != 1 ]; then
    echo "error: AKG_ALLOW_NON_GIT_SOURCE_FOR_TESTING must be 0 or 1" >&2
    exit 2
fi
if [ "$ALLOW_NON_GIT_SOURCE" = 0 ]; then
    if ! git -C "$PROJECT_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        echo "error: deployment source is not a Git worktree" >&2
        exit 2
    fi
    SOURCE_HEAD=$(git -C "$PROJECT_ROOT" rev-parse --verify HEAD)
    if [ "${#SOURCE_HEAD}" -ne 40 ]; then
        echo "error: deployment source HEAD is not a 40-character SHA-1" >&2
        exit 2
    fi
    case "$SOURCE_HEAD" in
        *[!0-9a-f]*)
            echo "error: deployment source HEAD is not a lowercase SHA-1" >&2
            exit 2
            ;;
    esac
    if [ -n "$(git -C "$PROJECT_ROOT" status --porcelain=v1 --untracked-files=all)" ]; then
        echo "error: deployment source worktree is not clean" >&2
        exit 2
    fi
fi

case "$SHARED_GROUP" in
    ""|*[!A-Za-z0-9_.-]*)
        echo "error: AKG_SHARED_GROUP contains unsupported characters" >&2
        exit 2
        ;;
esac
case "$VENV_OWNER" in
    ""|*[!A-Za-z0-9_.-]*)
        echo "error: AKG_VENV_OWNER contains unsupported characters" >&2
        exit 2
        ;;
esac
if ! command -v getent >/dev/null 2>&1 || ! getent group "$SHARED_GROUP" >/dev/null; then
    echo "error: shared deployment group does not exist: $SHARED_GROUP" >&2
    exit 2
fi
if ! getent passwd "$VENV_OWNER" >/dev/null; then
    echo "error: venv owner does not exist: $VENV_OWNER" >&2
    exit 2
fi

case "$VENV_DIR" in
    /*) ;;
    *)
        echo "error: AKG_VENV_DIR must be an absolute path" >&2
        exit 2
        ;;
esac

if [ -n "$CANN_ENV_FILE" ]; then
    case "$CANN_ENV_FILE" in
        /*) ;;
        *)
            echo "error: AKG_CANN_ENV_FILE must be an absolute path" >&2
            exit 2
            ;;
    esac
    if [ ! -r "$CANN_ENV_FILE" ]; then
        echo "error: CANN environment file is not readable: $CANN_ENV_FILE" >&2
        exit 2
    fi
    # Vendor set_env.sh is trusted platform configuration and must match the
    # installed torch_npu/Triton-Ascend stack.
    # shellcheck source=/dev/null
    . "$CANN_ENV_FILE"
fi

if ! command -v "$SYSTEM_PYTHON" >/dev/null 2>&1; then
    echo "error: Python executable was not found: $SYSTEM_PYTHON" >&2
    exit 2
fi
if ! "$SYSTEM_PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "error: Python 3.10 or newer is required: $SYSTEM_PYTHON" >&2
    exit 3
fi

if [ -e "$VENV_DIR" ]; then
    if [ -L "$VENV_DIR" ]; then
        echo "error: refusing a symlink AKG_VENV_DIR: $VENV_DIR" >&2
        exit 2
    fi
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        echo "error: existing AKG_VENV_DIR is not a usable venv: $VENV_DIR" >&2
        echo "refusing to delete or overwrite it" >&2
        exit 2
    fi
    echo "Reusing existing virtual environment: $VENV_DIR"
else
    echo "Creating system-site-packages virtual environment: $VENV_DIR"
    "$SYSTEM_PYTHON" -m venv --system-site-packages "$VENV_DIR"
fi
if ! "$VENV_DIR/bin/python" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "error: AKG_VENV_DIR must use Python 3.10 or newer: $VENV_DIR" >&2
    exit 3
fi

STACK_BEFORE=$("$VENV_DIR/bin/python" - <<'PY'
import importlib
import json

result = {}
for name in ("torch", "torch_npu", "triton"):
    module = importlib.import_module(name)
    result[name] = {
        "file": getattr(module, "__file__", None),
        "version": getattr(module, "__version__", None),
    }
import torch
result["npu_available"] = bool(torch.npu.is_available())
if not result["npu_available"]:
    raise RuntimeError("torch.npu.is_available() is false")
print(json.dumps(result, sort_keys=True))
PY
) || {
    echo "error: the preinstalled torch/torch_npu/Triton-Ascend stack is unavailable" >&2
    echo "install or repair the platform stack outside this project, then retry" >&2
    exit 3
}

if ! "$VENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
import re
from importlib.metadata import version

import setuptools
import wheel
import yaml

parts = tuple(int(item) for item in re.findall(r"\d+", version("setuptools"))[:2])
if parts < (69, 0):
    raise RuntimeError("setuptools 69 or newer is required")
PY
then
    if [ -n "${AKG_PURE_PYTHON_WHEELHOUSE:-}" ]; then
        if [ ! -d "$AKG_PURE_PYTHON_WHEELHOUSE" ]; then
            echo "error: AKG_PURE_PYTHON_WHEELHOUSE is not a directory" >&2
            exit 3
        fi
        "$VENV_DIR/bin/python" -m pip install \
            --no-index \
            --find-links "$AKG_PURE_PYTHON_WHEELHOUSE" \
            --only-binary=:all: \
            --no-deps \
            'setuptools>=69' \
            wheel \
            'PyYAML>=6.0.1,<7'
    elif [ "${AKG_ALLOW_NETWORK_INSTALL:-0}" = 1 ]; then
        "$VENV_DIR/bin/python" -m pip install \
            --only-binary=:all: \
            --no-deps \
            'setuptools>=69' \
            wheel \
            'PyYAML>=6.0.1,<7'
    else
        echo "error: PyYAML, wheel, or setuptools>=69 is missing" >&2
        echo "provide AKG_PURE_PYTHON_WHEELHOUSE or set AKG_ALLOW_NETWORK_INSTALL=1" >&2
        echo "no accelerator package was changed" >&2
        exit 3
    fi
fi

echo "Installing Ascend Kernel Lab without dependency resolution"
"$VENV_DIR/bin/python" -m pip install \
    --no-deps \
    --no-build-isolation \
    --force-reinstall \
    "$PROJECT_ROOT"

# A shared deployment venv is executable/readable by both service UIDs but is
# never writable by their shared group. Ownership changes are deliberately
# fail-closed: the installer must be run by a deployment administrator.
if ! find "$VENV_DIR" -xdev -type d \
        -exec chown "$VENV_OWNER:$SHARED_GROUP" '{}' + || \
   ! find "$VENV_DIR" -xdev -type f \
        -exec chown "$VENV_OWNER:$SHARED_GROUP" '{}' + || \
   ! find "$VENV_DIR" -xdev -type l \
        -exec chown -h "$VENV_OWNER:$SHARED_GROUP" '{}' +; then
    echo "error: could not assign the shared venv ownership" >&2
    exit 3
fi
find "$VENV_DIR" -xdev -type d -exec chmod 0755 '{}' +
find "$VENV_DIR" -xdev -type f -perm -0100 -exec chmod 0755 '{}' +
find "$VENV_DIR" -xdev -type f ! -perm -0100 -exec chmod 0644 '{}' +
if find "$VENV_DIR" -xdev \( -type d -o -type f \) \
    -perm -0022 -print -quit | grep . >/dev/null 2>&1; then
    echo "error: shared venv contains group-writable paths" >&2
    exit 3
fi
if find "$VENV_DIR" -xdev \( -type d -o -type f \) \
    \( ! -user "$VENV_OWNER" -o ! -group "$SHARED_GROUP" \) \
    -print -quit | grep . >/dev/null 2>&1; then
    echo "error: shared venv ownership verification failed" >&2
    exit 3
fi
if [ ! -x "$VENV_DIR/bin/python" ] || [ ! -x "$VENV_DIR/bin/akg" ]; then
    echo "error: shared venv entry points are not executable" >&2
    exit 3
fi

# The default repository configuration stores SQLite and artifacts here. Only
# the trusted controller/worker group can traverse or modify this state root;
# setgid preserves the group on every descendant created by either service UID.
if [ -e "$PROJECT_ROOT/runs" ]; then
    if [ -L "$PROJECT_ROOT/runs" ] || [ ! -d "$PROJECT_ROOT/runs" ]; then
        echo "error: runs path must be a real directory" >&2
        exit 3
    fi
    if ! chown "$VENV_OWNER:$SHARED_GROUP" "$PROJECT_ROOT/runs" || \
       ! chmod 2770 "$PROJECT_ROOT/runs"; then
        echo "error: could not repair the shared runs directory" >&2
        exit 3
    fi
elif ! install -d -m 2770 -o "$VENV_OWNER" -g "$SHARED_GROUP" "$PROJECT_ROOT/runs"; then
    echo "error: could not provision the shared runs directory" >&2
    exit 3
fi

STACK_AFTER=$("$VENV_DIR/bin/python" - <<'PY'
import importlib
import json

result = {}
for name in ("torch", "torch_npu", "triton"):
    module = importlib.import_module(name)
    result[name] = {
        "file": getattr(module, "__file__", None),
        "version": getattr(module, "__version__", None),
    }
import torch
result["npu_available"] = bool(torch.npu.is_available())
if not result["npu_available"]:
    raise RuntimeError("torch.npu.is_available() is false")
print(json.dumps(result, sort_keys=True))
PY
) || {
    echo "error: accelerator stack import failed after project installation" >&2
    exit 3
}

if [ "$STACK_BEFORE" != "$STACK_AFTER" ]; then
    echo "error: torch/torch_npu/triton identity changed during installation" >&2
    echo "before: $STACK_BEFORE" >&2
    echo "after:  $STACK_AFTER" >&2
    echo "stop and inspect this venv; the script will not delete it" >&2
    exit 4
fi

"$VENV_DIR/bin/python" -c 'import ascend_kernel_lab, yaml'
"$VENV_DIR/bin/akg" --help >/dev/null

if [ "$ALLOW_NON_GIT_SOURCE" = 0 ]; then
    if [ "$(git -C "$PROJECT_ROOT" rev-parse --verify HEAD)" != "$SOURCE_HEAD" ] || \
       [ -n "$(git -C "$PROJECT_ROOT" status --porcelain=v1 --untracked-files=all)" ]; then
        echo "error: deployment source changed during installation" >&2
        exit 4
    fi
fi

echo "Installation complete"
echo "  project: $PROJECT_ROOT"
echo "  venv:    $VENV_DIR"
echo "  stack:   $STACK_AFTER"
echo "Next: $VENV_DIR/bin/akg doctor -c $PROJECT_ROOT/configs/experiment_910c_kimi_k3.yaml"
