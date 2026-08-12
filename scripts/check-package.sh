#!/bin/sh
set -eu

umask 077

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON=${AKG_DEV_PYTHON:-}

if [ "$#" -gt 1 ]; then
    echo "usage: $0 [DIST_DIR]" >&2
    exit 2
fi

if [ -z "$PYTHON" ]; then
    if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
        PYTHON=$PROJECT_ROOT/.venv/bin/python
    else
        PYTHON=python3
    fi
fi
case "$PYTHON" in
    */*)
        PYTHON_DIR=$(CDPATH= cd -- "$(dirname -- "$PYTHON")" && pwd)
        PYTHON=$PYTHON_DIR/$(basename -- "$PYTHON")
        ;;
    *)
        PYTHON=$(command -v "$PYTHON")
        ;;
esac

DIST_DIR=${1:-"$PROJECT_ROOT/dist"}
if [ ! -d "$DIST_DIR" ]; then
    echo "error: distribution directory does not exist: $DIST_DIR" >&2
    exit 2
fi

WHEEL=
WHEEL_COUNT=0
for candidate in "$DIST_DIR"/*.whl; do
    [ -f "$candidate" ] || continue
    WHEEL=$candidate
    WHEEL_COUNT=$((WHEEL_COUNT + 1))
done
SDIST=
SDIST_COUNT=0
for candidate in "$DIST_DIR"/*.tar.gz; do
    [ -f "$candidate" ] || continue
    SDIST=$candidate
    SDIST_COUNT=$((SDIST_COUNT + 1))
done
if [ "$WHEEL_COUNT" -ne 1 ] || [ "$SDIST_COUNT" -ne 1 ]; then
    echo "error: expected exactly one wheel and one sdist in $DIST_DIR" >&2
    exit 2
fi

"$PYTHON" - "$WHEEL" "$SDIST" <<'PY'
from __future__ import annotations

import email.parser
import sys
import tarfile
import zipfile
from pathlib import PurePosixPath

wheel_path, sdist_path = sys.argv[1:]

with zipfile.ZipFile(wheel_path) as archive:
    wheel_name_list = archive.namelist()
    wheel_names = set(wheel_name_list)
    if len(wheel_names) != len(wheel_name_list):
        raise SystemExit("wheel contains duplicate members")
    unsafe_wheel = [
        name
        for name in wheel_names
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
    ]
    if unsafe_wheel:
        raise SystemExit(f"wheel contains unsafe member paths: {unsafe_wheel[:5]}")
    required_wheel = {
        "ascend_kernel_lab/__init__.py",
        "ascend_kernel_lab/__main__.py",
        "ascend_kernel_lab/cli.py",
    }
    missing = required_wheel - wheel_names
    if missing:
        raise SystemExit(f"wheel is missing runtime files: {sorted(missing)}")
    forbidden_roots = ("configs/", "deploy/", "docs/", "runs/", "scripts/", "task_specs/", "tests/")
    leaked = sorted(name for name in wheel_names if name.startswith(forbidden_roots))
    if leaked:
        raise SystemExit(f"wheel contains repository-external resources: {leaked[:5]}")
    metadata_names = [name for name in wheel_names if name.endswith(".dist-info/METADATA")]
    entry_names = [name for name in wheel_names if name.endswith(".dist-info/entry_points.txt")]
    if len(metadata_names) != 1 or len(entry_names) != 1:
        raise SystemExit("wheel must contain one METADATA and one entry_points.txt")
    metadata = email.parser.BytesParser().parsebytes(archive.read(metadata_names[0]))
    if metadata.get("Requires-Python") != ">=3.10":
        raise SystemExit("wheel Requires-Python contract changed")
    requirements = metadata.get_all("Requires-Dist", [])
    if not any(value.startswith("PyYAML") for value in requirements):
        raise SystemExit("wheel does not declare the PyYAML runtime dependency")
    entries = archive.read(entry_names[0]).decode("utf-8")
    if "akg = ascend_kernel_lab.cli:main" not in entries:
        raise SystemExit("wheel does not expose the akg console entry point")

with tarfile.open(sdist_path, "r:gz") as archive:
    members = archive.getmembers()
    names = {member.name for member in members}
    roots = {PurePosixPath(name).parts[0] for name in names if name}
    if len(roots) != 1:
        raise SystemExit("sdist must have exactly one top-level directory")
    root = next(iter(roots))
    required_sdist = {
        ".env.example",
        ".github/workflows/ci.yml",
        ".gitignore",
        "MANIFEST.in",
        "Makefile",
        "README.md",
        "configs/experiment_910c_kimi_k3.yaml",
        "deploy/env/controller.env.example",
        "docs/deployment-910c.md",
        "pyproject.toml",
        "runs/.gitkeep",
        "scripts/check-package.sh",
        "scripts/install-910c.sh",
        "scripts/validate-hidden-seed.py",
        "task_specs/k01_vector_add/task.yaml",
        "task_specs/k10_gemm_bias_gelu/hidden_template.json",
        "tests/integration/test_cli_e2e.py",
        "tests/fixtures/msprof/op_summary_v1.csv",
    }
    missing = sorted(f"{root}/{name}" for name in required_sdist if f"{root}/{name}" not in names)
    if missing:
        raise SystemExit(f"sdist is missing source/repository resources: {missing}")
    unsafe = []
    for member in members:
        relative = PurePosixPath(member.name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or member.issym()
            or member.islnk()
            or not (member.isfile() or member.isdir())
        ):
            unsafe.append(member.name)
        if "__pycache__" in relative.parts or member.name.endswith((".pyc", ".pyo")):
            unsafe.append(member.name)
        if len(relative.parts) >= 2 and relative.parts[1] == "runs":
            if relative.parts[2:] not in {(), (".gitkeep",)}:
                unsafe.append(member.name)
        basename = relative.name
        if basename == ".env" or basename.endswith(".env"):
            unsafe.append(member.name)
        if basename.endswith((".db", ".db-shm", ".db-wal", ".key", ".pem")):
            unsafe.append(member.name)
    if unsafe:
        raise SystemExit(f"sdist contains unsafe/generated members: {sorted(set(unsafe))[:10]}")
    shell_members = [member for member in members if "/scripts/" in member.name and member.name.endswith(".sh")]
    if not shell_members or any(member.mode & 0o111 == 0 for member in shell_members):
        raise SystemExit("every packaged shell entry point must be executable")
PY

TEMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/akg-package-check.XXXXXX")
cleanup() {
    rm -rf -- "$TEMP_ROOT"
}
trap cleanup EXIT HUP INT TERM

INSTALL_VENV=$TEMP_ROOT/venv
EXTRACT_DIR=$TEMP_ROOT/source
EMPTY_DIR=$TEMP_ROOT/empty
mkdir -p "$EXTRACT_DIR" "$EMPTY_DIR"

"$PYTHON" -m venv "$INSTALL_VENV"
PIP_NO_CACHE_DIR=1 "$INSTALL_VENV/bin/python" -m pip install \
    --disable-pip-version-check \
    --no-compile \
    --no-deps \
    "$WHEEL" >/dev/null
if [ ! -x "$INSTALL_VENV/bin/akg" ]; then
    echo "error: wheel installation did not generate the akg console script" >&2
    exit 2
fi
DEV_SITE=$(
    "$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'
)
WHEEL_SITE=$(
    "$INSTALL_VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'
)

"$PYTHON" - "$SDIST" "$EXTRACT_DIR" <<'PY'
from __future__ import annotations

import sys
import tarfile
from pathlib import Path, PurePosixPath

archive_path, destination = sys.argv[1:]
with tarfile.open(archive_path, "r:gz") as archive:
    for member in archive.getmembers():
        path = PurePosixPath(member.name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or member.issym()
            or member.islnk()
            or not (member.isfile() or member.isdir())
        ):
            raise SystemExit(f"refusing unsafe sdist member: {member.name}")
    if sys.version_info >= (3, 12):
        archive.extractall(Path(destination), filter="data")
    else:
        archive.extractall(Path(destination))
PY

CONFIG_PATH=
for candidate in "$EXTRACT_DIR"/*/configs/experiment_910c_kimi_k3.yaml; do
    [ -f "$candidate" ] || continue
    CONFIG_PATH=$candidate
done
if [ -z "$CONFIG_PATH" ]; then
    echo "error: cannot locate the extracted external experiment config" >&2
    exit 2
fi

(
    cd "$EMPTY_DIR"
    PYTHONPATH=$DEV_SITE "$INSTALL_VENV/bin/python" - "$WHEEL_SITE" "$CONFIG_PATH" <<'PY'
from __future__ import annotations

import sys
from importlib.metadata import version
from pathlib import Path

import ascend_kernel_lab
from ascend_kernel_lab.config import load_config
from ascend_kernel_lab.tasks import TaskRegistry

site = Path(sys.argv[1]).resolve()
module_path = Path(ascend_kernel_lab.__file__).resolve()
if not module_path.is_relative_to(site):
    raise SystemExit(f"wheel smoke imported the source tree instead of the wheel: {module_path}")
if version("ascend-kernel-lab") != ascend_kernel_lab.__version__:
    raise SystemExit("installed distribution and module versions differ")
config = load_config(sys.argv[2])
loaded = TaskRegistry(config.task_root).load_many(config.tasks)
if len(loaded) != 10:
    raise SystemExit("external task registry did not load all configured tasks")
if config.artifact_root != config.project_root / "runs":
    raise SystemExit("external artifact-root resolution contract changed")
PY
    PYTHONPATH=$DEV_SITE "$INSTALL_VENV/bin/python" -m ascend_kernel_lab --version >/dev/null
    PYTHONPATH=$DEV_SITE "$INSTALL_VENV/bin/python" -m ascend_kernel_lab --help >/dev/null
    PYTHONPATH=$DEV_SITE "$INSTALL_VENV/bin/akg" --version >/dev/null
    set +e
    PYTHONPATH=$DEV_SITE "$INSTALL_VENV/bin/akg" doctor >default-config.out 2>&1
    DEFAULT_STATUS=$?
    set -e
    if [ "$DEFAULT_STATUS" -ne 2 ]; then
        echo "error: wheel-only CLI did not fail clearly without the external config" >&2
        exit 2
    fi
    if ! grep -F "configuration file does not exist: configs/experiment_910c_kimi_k3.yaml" default-config.out >/dev/null; then
        echo "error: wheel-only CLI did not preserve the external-config path contract" >&2
        exit 2
    fi
    PYTHONPATH=$DEV_SITE "$INSTALL_VENV/bin/python" -m ascend_kernel_lab db upgrade -c "$CONFIG_PATH" >/dev/null
)

echo "Package checks passed"
