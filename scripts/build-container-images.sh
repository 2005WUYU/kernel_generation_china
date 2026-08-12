#!/bin/sh
set -eu

usage() {
    echo "usage: WORKER_BASE=<immutable-ref> CONTROLLER_BASE=<immutable-ref> CONTROLLER_NODE_BASE=<immutable-ref> CLAUDE_CODE_VERSION=<exact-version> CLAUDE_CODE_INTEGRITY=<sha512-value> $0" >&2
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
CONTROLLER_NODE_BASE=${CONTROLLER_NODE_BASE:-}
CLAUDE_CODE_VERSION=${CLAUDE_CODE_VERSION:-}
CLAUDE_CODE_INTEGRITY=${CLAUDE_CODE_INTEGRITY:-}
WORKER_IMAGE=${AKG_WORKER_IMAGE:-ascend-kernel-lab-worker:local}
CONTROLLER_IMAGE=${AKG_CONTROLLER_IMAGE:-ascend-kernel-lab-controller:local}

if ! printf '%s\n' "$CLAUDE_CODE_VERSION" | \
        grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)?$'; then
    echo "error: CLAUDE_CODE_VERSION must be one exact semantic version" >&2
    exit 2
fi
case "$CLAUDE_CODE_INTEGRITY" in
    sha512-?*) ;;
    *) echo "error: CLAUDE_CODE_INTEGRITY must be the approved sha512 integrity value" >&2; exit 2 ;;
esac
case "${CLAUDE_CODE_INTEGRITY#sha512-}" in
    *[!0-9A-Za-z+/=]*)
        echo "error: CLAUDE_CODE_INTEGRITY contains non-base64 characters" >&2
        exit 2
        ;;
esac

NODE_LOCAL_BASE_TAG=
case "$CONTROLLER_NODE_BASE" in
    sha256:????????????????????????????????????????????????????????????????)
        case "${CONTROLLER_NODE_BASE#sha256:}" in
            *[!0-9a-f]*)
                echo "error: local Node image ID must be 64 lowercase hex characters" >&2
                exit 2
                ;;
        esac
        INSPECTED_ID=$(docker image inspect --format '{{.Id}}' "$CONTROLLER_NODE_BASE" 2>/dev/null || true)
        if [ "$INSPECTED_ID" != "$CONTROLLER_NODE_BASE" ]; then
            echo "error: CONTROLLER_NODE_BASE does not resolve to the exact local full image ID" >&2
            exit 3
        fi
        NODE_LOCAL_BASE_TAG="ascend-kernel-lab-node-base:${CONTROLLER_NODE_BASE#sha256:}"
        docker image tag "$CONTROLLER_NODE_BASE" "$NODE_LOCAL_BASE_TAG"
        NODE_BUILD_BASE=$NODE_LOCAL_BASE_TAG
        ;;
    *@sha256:????????????????????????????????????????????????????????????????)
        case "${CONTROLLER_NODE_BASE##*@sha256:}" in
            *[!0-9a-f]*)
                echo "error: Node registry digest must be 64 lowercase hex characters" >&2
                exit 2
                ;;
        esac
        NODE_BUILD_BASE=$CONTROLLER_NODE_BASE
        ;;
    *)
        echo "error: CONTROLLER_NODE_BASE must be repo@sha256:digest or a full local sha256 image ID" >&2
        exit 2
        ;;
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
NODE_OS=$(docker image inspect --format '{{.Os}}' "$NODE_BUILD_BASE" 2>/dev/null || true)
NODE_ARCH=$(docker image inspect --format '{{.Architecture}}' "$NODE_BUILD_BASE" 2>/dev/null || true)
if [ "$NODE_OS:$NODE_ARCH" != "linux:arm64" ]; then
    echo "error: Node base must already exist locally as linux/arm64 (found $NODE_OS/$NODE_ARCH)" >&2
    exit 3
fi

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)

docker build --pull=false \
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

docker build --pull=false \
    --build-arg "CONTROLLER_BASE=$CONTROLLER_BUILD_BASE" \
    --build-arg "CONTROLLER_NODE_BASE=$NODE_BUILD_BASE" \
    --build-arg "CLAUDE_CODE_VERSION=$CLAUDE_CODE_VERSION" \
    --build-arg "CLAUDE_CODE_INTEGRITY=$CLAUDE_CODE_INTEGRITY" \
    --label "org.ascend-kernel-lab.controller-base=$CONTROLLER_BASE" \
    --label "org.ascend-kernel-lab.controller-node-base=$CONTROLLER_NODE_BASE" \
    --label "org.ascend-kernel-lab.claude-code-version=$CLAUDE_CODE_VERSION" \
    --label "org.ascend-kernel-lab.claude-code-integrity=$CLAUDE_CODE_INTEGRITY" \
    --file "$PROJECT_ROOT/deploy/container/Dockerfile.controller" \
    --tag "$CONTROLLER_IMAGE" \
    "$PROJECT_ROOT/deploy/container"

if [ -n "$CONTROLLER_LOCAL_BASE_TAG" ]; then
    CURRENT_ID=$(docker image inspect --format '{{.Id}}' "$CONTROLLER_LOCAL_BASE_TAG")
    if [ "$CURRENT_ID" != "$CONTROLLER_BASE" ]; then
        echo "error: local Controller base identity changed during build" >&2
        exit 4
    fi
fi
if [ -n "$NODE_LOCAL_BASE_TAG" ]; then
    CURRENT_ID=$(docker image inspect --format '{{.Id}}' "$NODE_LOCAL_BASE_TAG")
    if [ "$CURRENT_ID" != "$CONTROLLER_NODE_BASE" ]; then
        echo "error: local Node base identity changed during build" >&2
        exit 4
    fi
fi

docker image inspect --format 'built {{.RepoTags}} {{.Id}} {{.Os}}/{{.Architecture}}' \
    "$WORKER_IMAGE" "$CONTROLLER_IMAGE"
