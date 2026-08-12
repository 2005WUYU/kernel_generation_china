#!/bin/sh
set -eu

usage() {
    echo "usage: WORKER_BASE=<immutable-ref> CONTROLLER_BASE=<immutable-ref> CLAUDE_CODE_ARCHIVE=<absolute-path> CLAUDE_CODE_VERSION=<exact-version> CLAUDE_CODE_SHA256=<64hex> $0" >&2
}

if [ "$#" -ne 0 ]; then
    usage
    exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker is required" >&2
    exit 2
fi

WORKER_BASE=${WORKER_BASE:-}
CONTROLLER_BASE=${CONTROLLER_BASE:-}
CLAUDE_CODE_ARCHIVE=${CLAUDE_CODE_ARCHIVE:-}
CLAUDE_CODE_VERSION=${CLAUDE_CODE_VERSION:-}
CLAUDE_CODE_SHA256=${CLAUDE_CODE_SHA256:-}
WORKER_IMAGE=${AKG_WORKER_IMAGE:-ascend-kernel-lab-worker:local}
CONTROLLER_IMAGE=${AKG_CONTROLLER_IMAGE:-ascend-kernel-lab-controller:local}

if ! printf '%s\n' "$CLAUDE_CODE_VERSION" | \
        grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)?$'; then
    echo "error: CLAUDE_CODE_VERSION must be one exact semantic version" >&2
    exit 2
fi
case "$CLAUDE_CODE_SHA256" in
    ????????????????????????????????????????????????????????????????) ;;
    *) echo "error: CLAUDE_CODE_SHA256 must be exactly 64 lowercase hex characters" >&2; exit 2 ;;
esac
case "$CLAUDE_CODE_SHA256" in
    *[!0-9a-f]*) echo "error: CLAUDE_CODE_SHA256 must be lowercase hexadecimal" >&2; exit 2 ;;
esac
case "$CLAUDE_CODE_ARCHIVE" in
    /*) ;;
    *) echo "error: CLAUDE_CODE_ARCHIVE must be an absolute path" >&2; exit 2 ;;
esac
if [ ! -f "$CLAUDE_CODE_ARCHIVE" ] || [ -L "$CLAUDE_CODE_ARCHIVE" ] || [ ! -r "$CLAUDE_CODE_ARCHIVE" ]; then
    echo "error: CLAUDE_CODE_ARCHIVE must be a readable regular non-symlink file" >&2
    exit 3
fi
if ! command -v sha256sum >/dev/null 2>&1; then
    echo "error: sha256sum is required" >&2
    exit 2
fi
ARCHIVE_ACTUAL_SHA=$(sha256sum "$CLAUDE_CODE_ARCHIVE" | awk '{print $1}')
if [ "$ARCHIVE_ACTUAL_SHA" != "$CLAUDE_CODE_SHA256" ]; then
    echo "error: Claude Code archive SHA-256 mismatch" >&2
    exit 3
fi
ARCHIVE_DIR=$(CDPATH= cd -- "$(dirname -- "$CLAUDE_CODE_ARCHIVE")" && pwd -P)
ARCHIVE_NAME=$(basename -- "$CLAUDE_CODE_ARCHIVE")
case "$ARCHIVE_NAME" in
    ""|.*|*[!0-9A-Za-z._-]*) echo "error: Claude Code archive filename is unsafe" >&2; exit 2 ;;
esac

WORKER_LOCAL_BASE_TAG=
case "$WORKER_BASE" in
    sha256:????????????????????????????????????????????????????????????????)
        case "${WORKER_BASE#sha256:}" in
            *[!0-9a-f]*)
                echo "error: local Worker image ID must be 64 lowercase hex characters" >&2
                exit 2
                ;;
        esac
        INSPECTED_ID=$(docker image inspect --format '{{.Id}}' "$WORKER_BASE" 2>/dev/null || true)
        if [ "$INSPECTED_ID" != "$WORKER_BASE" ]; then
            echo "error: WORKER_BASE does not resolve to the exact local full image ID" >&2
            exit 3
        fi
        WORKER_LOCAL_BASE_TAG="ascend-kernel-lab-worker-base:${WORKER_BASE#sha256:}"
        docker image tag "$WORKER_BASE" "$WORKER_LOCAL_BASE_TAG"
        WORKER_BUILD_BASE=$WORKER_LOCAL_BASE_TAG
        ;;
    *@sha256:????????????????????????????????????????????????????????????????)
        case "${WORKER_BASE##*@sha256:}" in
            *[!0-9a-f]*)
                echo "error: Worker registry digest must be 64 lowercase hex characters" >&2
                exit 2
                ;;
        esac
        WORKER_BUILD_BASE=$WORKER_BASE
        ;;
    *)
        echo "error: WORKER_BASE must be repo@sha256:digest or a full local sha256 image ID" >&2
        exit 2
        ;;
esac

CONTROLLER_LOCAL_BASE_TAG=
case "$CONTROLLER_BASE" in
    sha256:????????????????????????????????????????????????????????????????)
        case "${CONTROLLER_BASE#sha256:}" in
            *[!0-9a-f]*)
                echo "error: local Controller image ID must be 64 lowercase hex characters" >&2
                exit 2
                ;;
        esac
        INSPECTED_ID=$(docker image inspect --format '{{.Id}}' "$CONTROLLER_BASE" 2>/dev/null || true)
        if [ "$INSPECTED_ID" != "$CONTROLLER_BASE" ]; then
            echo "error: CONTROLLER_BASE does not resolve to the exact local full image ID" >&2
            exit 3
        fi
        CONTROLLER_LOCAL_BASE_TAG="ascend-kernel-lab-controller-base:${CONTROLLER_BASE#sha256:}"
        docker image tag "$CONTROLLER_BASE" "$CONTROLLER_LOCAL_BASE_TAG"
        CONTROLLER_BUILD_BASE=$CONTROLLER_LOCAL_BASE_TAG
        ;;
    *@sha256:????????????????????????????????????????????????????????????????)
        case "${CONTROLLER_BASE##*@sha256:}" in
            *[!0-9a-f]*)
                echo "error: Controller registry digest must be 64 lowercase hex characters" >&2
                exit 2
                ;;
        esac
        CONTROLLER_BUILD_BASE=$CONTROLLER_BASE
        ;;
    *)
        echo "error: CONTROLLER_BASE must be repo@sha256:digest or a full local sha256 image ID" >&2
        exit 2
        ;;
esac

WORKER_OS=$(docker image inspect --format '{{.Os}}' "$WORKER_BUILD_BASE" 2>/dev/null || true)
WORKER_ARCH=$(docker image inspect --format '{{.Architecture}}' "$WORKER_BUILD_BASE" 2>/dev/null || true)
if [ "$WORKER_OS:$WORKER_ARCH" != "linux:arm64" ]; then
    echo "error: Worker base must already exist locally as linux/arm64 (found $WORKER_OS/$WORKER_ARCH)" >&2
    exit 3
fi
CONTROLLER_OS=$(docker image inspect --format '{{.Os}}' "$CONTROLLER_BUILD_BASE" 2>/dev/null || true)
CONTROLLER_ARCH=$(docker image inspect --format '{{.Architecture}}' "$CONTROLLER_BUILD_BASE" 2>/dev/null || true)
if [ "$CONTROLLER_OS:$CONTROLLER_ARCH" != "linux:arm64" ]; then
    echo "error: Controller base must already exist locally as linux/arm64 (found $CONTROLLER_OS/$CONTROLLER_ARCH)" >&2
    exit 3
fi
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

docker build --pull=false \
    --network=none \
    --build-arg "WORKER_BASE=$WORKER_BUILD_BASE" \
    --label "org.ascend-kernel-lab.worker-base=$WORKER_BASE" \
    --file "$PROJECT_ROOT/deploy/container/Dockerfile.worker" \
    --tag "$WORKER_IMAGE" \
    "$PROJECT_ROOT/deploy/container"

if [ -n "$WORKER_LOCAL_BASE_TAG" ]; then
    CURRENT_ID=$(docker image inspect --format '{{.Id}}' "$WORKER_LOCAL_BASE_TAG")
    if [ "$CURRENT_ID" != "$WORKER_BASE" ]; then
        echo "error: local Worker base identity changed during build" >&2
        exit 4
    fi
fi

tar -cf - \
    -C "$PROJECT_ROOT/deploy/container" Dockerfile.controller controller-entrypoint.sh \
    -C "$ARCHIVE_DIR" "$ARCHIVE_NAME" | \
    docker build --pull=false \
        --network=none \
        --build-arg "CONTROLLER_BASE=$CONTROLLER_BUILD_BASE" \
        --build-arg "CLAUDE_CODE_ARCHIVE_NAME=$ARCHIVE_NAME" \
        --build-arg "CLAUDE_CODE_VERSION=$CLAUDE_CODE_VERSION" \
        --build-arg "CLAUDE_CODE_SHA256=$CLAUDE_CODE_SHA256" \
        --label "org.ascend-kernel-lab.controller-base=$CONTROLLER_BASE" \
        --label "org.ascend-kernel-lab.claude-code-version=$CLAUDE_CODE_VERSION" \
        --label "org.ascend-kernel-lab.claude-code-sha256=$CLAUDE_CODE_SHA256" \
        --file Dockerfile.controller \
        --tag "$CONTROLLER_IMAGE" \
        -

if [ -n "$CONTROLLER_LOCAL_BASE_TAG" ]; then
    CURRENT_ID=$(docker image inspect --format '{{.Id}}' "$CONTROLLER_LOCAL_BASE_TAG")
    if [ "$CURRENT_ID" != "$CONTROLLER_BASE" ]; then
        echo "error: local Controller base identity changed during build" >&2
        exit 4
    fi
fi
docker image inspect --format 'built {{.RepoTags}} {{.Id}} {{.Os}}/{{.Architecture}}' \
    "$WORKER_IMAGE" "$CONTROLLER_IMAGE"
