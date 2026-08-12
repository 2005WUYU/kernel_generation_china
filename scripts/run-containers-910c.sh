#!/bin/sh
set -eu

usage() {
    echo "usage: $0 init|probe|baseline|start-worker|start-controller|status|stop" >&2
}

if [ "$#" -ne 1 ]; then
    usage
    exit 2
fi
ACTION=$1
case "$ACTION" in
    init|probe|baseline|start-worker|start-controller|status|stop) ;;
    *) usage; exit 2 ;;
esac

if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker is required" >&2
    exit 2
fi

PROJECT_ROOT=${AKG_PROJECT_ROOT:-}
CONTROLLER_ENV_FILE=${AKG_CONTROLLER_ENV_FILE:-/etc/ascend-kernel-lab/controller.env}
HIDDEN_ENV_FILE=${AKG_HIDDEN_ENV_FILE:-/etc/ascend-kernel-lab/hidden.env}
WORKER_ENV_FILE=${AKG_WORKER_ENV_FILE:-/etc/ascend-kernel-lab/worker-container.env}
WORKER_IMAGE=${AKG_WORKER_IMAGE:-ascend-kernel-lab-worker:local}
CONTROLLER_IMAGE=${AKG_CONTROLLER_IMAGE:-ascend-kernel-lab-controller:local}
WORKER_NAME=${AKG_WORKER_CONTAINER:-ascend-kernel-worker}
CONTROLLER_NAME=${AKG_CONTROLLER_CONTAINER:-ascend-kernel-controller}
DEVICE_ID=${AKG_DEVICE_ID:-0}
TASK_ID=${AKG_TASK_ID:-}
CONTROLLER_UID=${AKG_CONTROLLER_UID:-}
WORKER_UID=${AKG_WORKER_UID:-}
SHARED_GID=${AKG_SHARED_GID:-}
NPU_DEVICE_GID=${AKG_NPU_DEVICE_GID:-}
CONTAINER_PROJECT_ROOT=/workspace
LOCK_ROOT=${AKG_DEVICE_LOCK_ROOT:-/var/lock/ascend-kernel-lab}

# Status and emergency stop must remain available even if the checkout or its
# permissions are damaged. They deliberately require no environment files.
case "$ACTION" in
    status)
        docker container inspect --format '{{.Name}} running={{.State.Running}} exit={{.State.ExitCode}} image={{.Image}}' \
            "$WORKER_NAME" "$CONTROLLER_NAME" 2>/dev/null || true
        exit 0
        ;;
    stop)
        for name in "$CONTROLLER_NAME" "$WORKER_NAME"; do
            if docker container inspect "$name" >/dev/null 2>&1; then
                docker container stop --time 45 "$name" >/dev/null
                docker container rm "$name" >/dev/null
                echo "stopped and removed $name"
            fi
        done
        exit 0
        ;;
esac

case "$DEVICE_ID" in
    ""|*','*|*[!0-9]*)
        echo "error: AKG_DEVICE_ID must select exactly one numeric NPU" >&2
        exit 2
        ;;
esac
case "$TASK_ID" in
    ""|k[0-9][0-9]_[a-z0-9_]*) ;;
    *) echo "error: AKG_TASK_ID must be empty or a task ID such as k01_vector_add" >&2; exit 2 ;;
esac
for identity in "$CONTROLLER_UID" "$WORKER_UID"; do
    case "$identity" in
        ""|*[!0-9]*)
            echo "error: AKG_CONTROLLER_UID and AKG_WORKER_UID must be numeric IDs" >&2
            exit 2
            ;;
    esac
done
case "$SHARED_GID" in
    ""|0|*[!0-9]*)
        echo "error: AKG_SHARED_GID must be a non-zero numeric ID" >&2
        exit 2
        ;;
esac
case "$NPU_DEVICE_GID" in
    ""|*[!0-9]*)
        echo "error: AKG_NPU_DEVICE_GID must be the numeric group owning NPU devices" >&2
        exit 2
        ;;
esac
case "$LOCK_ROOT" in
    /*) ;;
    *) echo "error: AKG_DEVICE_LOCK_ROOT must be absolute" >&2; exit 2 ;;
esac
if [ -L "$LOCK_ROOT" ] || [ ! -d "$LOCK_ROOT" ]; then
    echo "error: AKG_DEVICE_LOCK_ROOT must be a real pre-created directory" >&2
    exit 3
fi
LOCK_GID=$(stat -c '%g' "$LOCK_ROOT")
LOCK_MODE=$(stat -c '%a' "$LOCK_ROOT")
if [ "$LOCK_GID" != "$SHARED_GID" ] || [ "$LOCK_MODE" != 2770 ]; then
    echo "error: device lock root must have AKG_SHARED_GID and exact mode 2770" >&2
    exit 3
fi

case "$PROJECT_ROOT" in
    /*) ;;
    *) echo "error: AKG_PROJECT_ROOT must be the absolute path of the remote clean clone" >&2; exit 2 ;;
esac
PROJECT_ROOT=$(CDPATH= cd -- "$PROJECT_ROOT" && pwd -P)

if [ ! -d "$PROJECT_ROOT/.git" ]; then
    echo "error: AKG_PROJECT_ROOT must be a complete Git clone" >&2
    exit 2
fi
TOPLEVEL=$(git -C "$PROJECT_ROOT" rev-parse --show-toplevel 2>/dev/null || true)
HEAD=$(git -C "$PROJECT_ROOT" rev-parse --verify HEAD 2>/dev/null || true)
if [ "$TOPLEVEL" != "$PROJECT_ROOT" ]; then
    echo "error: AKG_PROJECT_ROOT must be the exact Git top-level" >&2
    exit 3
fi
case "$HEAD" in
    ????????????????????????????????????????|????????????????????????????????????????????????????????????????) ;;
    *) echo "error: remote clone must have a full Git commit" >&2; exit 3 ;;
esac
case "$HEAD" in
    *[!0-9a-f]*) echo "error: Git commit must be lowercase hexadecimal" >&2; exit 3 ;;
esac
if [ -n "$(git -C "$PROJECT_ROOT" status --porcelain=v1 --untracked-files=all)" ]; then
    echo "error: remote clone must be clean before container startup" >&2
    exit 3
fi
if [ -L "$PROJECT_ROOT/runs" ] || [ ! -d "$PROJECT_ROOT/runs" ]; then
    echo "error: $PROJECT_ROOT/runs must be a real directory" >&2
    exit 3
fi
RUNS_GID=$(stat -c '%g' "$PROJECT_ROOT/runs")
RUNS_MODE=$(stat -c '%a' "$PROJECT_ROOT/runs")
if [ "$RUNS_GID" != "$SHARED_GID" ]; then
    echo "error: runs group ID $RUNS_GID does not match AKG_SHARED_GID=$SHARED_GID" >&2
    exit 3
fi
case "$RUNS_MODE" in
    2770) ;;
    *) echo "error: runs must have exact mode 2770 (found $RUNS_MODE)" >&2; exit 3 ;;
esac

validate_env_file() {
    file=$1
    label=$2
    if [ ! -f "$file" ] || [ -L "$file" ]; then
        echo "error: $label environment file is missing or unsafe: $file" >&2
        exit 3
    fi
    if grep -Ev '^[[:space:]]*(#.*)?$|^[A-Za-z_][A-Za-z0-9_]*=.*$' "$file" >/dev/null; then
        echo "error: $label environment file has invalid Docker env-file syntax" >&2
        exit 3
    fi
    mode=$(stat -c '%a' "$file")
    owner=$(stat -c '%u' "$file")
    if [ "$owner" != 0 ]; then
        echo "error: $label environment file must be owned by root: $file" >&2
        exit 3
    fi
    case "$mode" in
        600|400) ;;
        *) echo "error: $label environment file must have mode 0600 or 0400: $file" >&2; exit 3 ;;
    esac
}

reject_worker_model_env_file() {
    if grep -Eiq '^[[:space:]]*[A-Za-z_][A-Za-z0-9_]*(API_KEY|AUTH|CREDENTIAL|PASSWORD|PRIVATE_KEY|PROXY|SECRET|TOKEN)[A-Za-z0-9_]*[[:space:]]*=' "$WORKER_ENV_FILE"; then
        echo "error: Worker environment file contains model/provider settings" >&2
        exit 10
    fi
    if grep -Eq '^[[:space:]]*AKG_HIDDEN_SEED[[:space:]]*=' "$WORKER_ENV_FILE"; then
        echo "error: Worker environment file must not contain the hidden seed" >&2
        exit 10
    fi
}

validate_secret_separation() {
    if grep -Eq '^[[:space:]]*AKG_HIDDEN_SEED[[:space:]]*=' "$CONTROLLER_ENV_FILE"; then
        echo "error: controller.env must not contain AKG_HIDDEN_SEED" >&2
        exit 10
    fi
}

reject_hidden_model_env_file() {
    if grep -Eq '^[[:space:]]*(ANTHROPIC_|KIMI_|OPENAI_|AIPING_)' "$HIDDEN_ENV_FILE"; then
        echo "error: hidden.env must not contain model/provider settings" >&2
        exit 10
    fi
}

ensure_absent() {
    name=$1
    if docker container inspect "$name" >/dev/null 2>&1; then
        echo "error: container already exists: $name; run '$0 stop' first" >&2
        exit 3
    fi
}

worker_library_path() {
    # Ascend Docker Runtime bind-mounts the host driver, but older runtime
    # releases do not consistently extend LD_LIBRARY_PATH for an arbitrary
    # non-root UID when the image entrypoint is overridden.  Keep the
    # platform image's own CANN paths and add the standard driver locations.
    image_path=$(docker image inspect --format '{{range .Config.Env}}{{println .}}{{end}}' \
        "$WORKER_IMAGE" | sed -n 's/^LD_LIBRARY_PATH=//p' | tail -n 1)
    driver_path=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver
    if [ -n "$image_path" ]; then
        printf '%s:%s\n' "$driver_path" "$image_path"
    else
        printf '%s\n' "$driver_path"
    fi
}

case "$ACTION" in
    init)
        docker run --rm \
            --runtime=runc \
            --network none \
            --read-only \
            --user "$CONTROLLER_UID:$SHARED_GID" \
            --cap-drop ALL \
            --security-opt no-new-privileges \
            --pids-limit 128 \
            --memory "${AKG_INIT_MEMORY:-2g}" \
            --tmpfs /tmp:rw,nosuid,nodev,noexec,mode=1777 \
            --volume "$PROJECT_ROOT:$CONTAINER_PROJECT_ROOT:ro" \
            --volume "$PROJECT_ROOT/runs:$CONTAINER_PROJECT_ROOT/runs:rw" \
            --entrypoint /bin/sh \
            "$CONTROLLER_IMAGE" -c \
            'umask 0007; export PYTHONPATH=/workspace/src; cd /workspace; python3 -m ascend_kernel_lab db upgrade -c configs/experiment_910c_kimi_k3.yaml'
        ;;
    probe|baseline)
        WORKER_LD_LIBRARY_PATH=$(worker_library_path)
        if docker container inspect "$WORKER_NAME" --format '{{.State.Running}}' 2>/dev/null | grep -qx true; then
            echo "error: stop the project Worker before probe or baseline maintenance" >&2
            exit 3
        fi
        validate_env_file "$WORKER_ENV_FILE" worker
        reject_worker_model_env_file
        if [ "$ACTION" = probe ]; then
            MAINTENANCE_COMMAND='python3 -m ascend_kernel_lab probe all -c configs/experiment_910c_kimi_k3.yaml -o runs/probe'
        else
            MAINTENANCE_COMMAND='python3 -m ascend_kernel_lab baseline run -c configs/experiment_910c_kimi_k3.yaml'
            if [ -n "$TASK_ID" ]; then
                MAINTENANCE_COMMAND="$MAINTENANCE_COMMAND --task $TASK_ID"
            fi
        fi
        docker run --rm \
            --runtime=ascend \
            --network none \
            --read-only \
            --user "$WORKER_UID:$SHARED_GID" \
            --group-add "$NPU_DEVICE_GID" \
            --cap-drop ALL \
            --security-opt no-new-privileges \
            --pids-limit 1024 \
            --memory "${AKG_WORKER_MEMORY:-48g}" \
            --tmpfs /tmp:rw,nosuid,nodev,exec,mode=1777 \
            --env-file "$WORKER_ENV_FILE" \
            --env "LD_LIBRARY_PATH=$WORKER_LD_LIBRARY_PATH" \
            --env "ASCEND_VISIBLE_DEVICES=$DEVICE_ID" \
            --env DEVICE_ID=0 \
            --env HOME=/tmp/akg-home \
            --env TRITON_CACHE_DIR=/tmp/akg-triton-cache \
            --env TRITON_DUMP_DIR=/tmp/akg-triton-dump \
            --env AKG_DEVICE_LOCK_ROOT=/var/lock/ascend-kernel-lab \
            --env GIT_CONFIG_COUNT=1 \
            --env GIT_CONFIG_KEY_0=safe.directory \
            --env "GIT_CONFIG_VALUE_0=$CONTAINER_PROJECT_ROOT" \
            --env GIT_OPTIONAL_LOCKS=0 \
            --volume "$PROJECT_ROOT:$CONTAINER_PROJECT_ROOT:ro" \
            --volume "$PROJECT_ROOT/runs:$CONTAINER_PROJECT_ROOT/runs:rw" \
            --volume "$LOCK_ROOT:/var/lock/ascend-kernel-lab:rw" \
            --entrypoint /bin/sh \
            "$WORKER_IMAGE" -c \
            'set -eu
             umask 0007
             mkdir -p "$HOME"
             mkdir -p "$TRITON_CACHE_DIR" "$TRITON_DUMP_DIR"
             chmod 0700 "$HOME"
             chmod 0700 "$TRITON_CACHE_DIR" "$TRITON_DUMP_DIR"
             export PYTHONPATH=/workspace/src
             cd /workspace
             python3 -c '\''import torch, torch_npu, triton; assert torch.npu.is_available(); assert torch.npu.device_count() == 1'\''
             exec /bin/sh -c "$1"' sh "$MAINTENANCE_COMMAND"
        ;;
    start-worker)
        WORKER_LD_LIBRARY_PATH=$(worker_library_path)
        validate_env_file "$HIDDEN_ENV_FILE" hidden
        validate_env_file "$WORKER_ENV_FILE" worker
        reject_hidden_model_env_file
        reject_worker_model_env_file
        ensure_absent "$WORKER_NAME"
        docker run --detach \
            --name "$WORKER_NAME" \
            --user "$WORKER_UID:$SHARED_GID" \
            --group-add "$NPU_DEVICE_GID" \
            --restart on-failure:3 \
            --health-cmd 'kill -0 1' \
            --health-interval 10s \
            --health-timeout 3s \
            --health-retries 3 \
            --health-start-period 5s \
            --runtime=ascend \
            --network none \
            --read-only \
            --cap-drop ALL \
            --security-opt no-new-privileges \
            --pids-limit 1024 \
            --memory "${AKG_WORKER_MEMORY:-48g}" \
            --log-opt "max-size=${AKG_LOG_MAX_SIZE:-20m}" \
            --log-opt "max-file=${AKG_LOG_MAX_FILES:-5}" \
            --tmpfs /tmp:rw,nosuid,nodev,noexec,mode=1777 \
            --env-file "$WORKER_ENV_FILE" \
            --env-file "$HIDDEN_ENV_FILE" \
            --env "LD_LIBRARY_PATH=$WORKER_LD_LIBRARY_PATH" \
            --env "ASCEND_VISIBLE_DEVICES=$DEVICE_ID" \
            --env "DEVICE_ID=0" \
            --env "AKG_PROJECT_ROOT=$CONTAINER_PROJECT_ROOT" \
            --env "AKG_CONFIG_PATH=$CONTAINER_PROJECT_ROOT/configs/experiment_910c_kimi_k3.yaml" \
            --env AKG_DEVICE_LOCK_ROOT=/var/lock/ascend-kernel-lab \
            --env HOME=/tmp/akg-home \
            --env GIT_CONFIG_COUNT=1 \
            --env GIT_CONFIG_KEY_0=safe.directory \
            --env "GIT_CONFIG_VALUE_0=$CONTAINER_PROJECT_ROOT" \
            --env GIT_OPTIONAL_LOCKS=0 \
            --volume "$PROJECT_ROOT:$CONTAINER_PROJECT_ROOT:ro" \
            --volume "$PROJECT_ROOT/runs:$CONTAINER_PROJECT_ROOT/runs:rw" \
            --volume "$LOCK_ROOT:/var/lock/ascend-kernel-lab:rw" \
            "$WORKER_IMAGE"
        ;;
    start-controller)
        validate_env_file "$HIDDEN_ENV_FILE" hidden
        validate_env_file "$CONTROLLER_ENV_FILE" controller
        reject_hidden_model_env_file
        validate_secret_separation
        if ! docker container inspect "$WORKER_NAME" --format '{{.State.Running}} {{.State.Health.Status}}' 2>/dev/null | grep -qx 'true healthy'; then
            echo "error: Worker must be healthy before controller startup" >&2
            exit 3
        fi
        ensure_absent "$CONTROLLER_NAME"
        if [ -n "$TASK_ID" ]; then
            set -- python3 -m ascend_kernel_lab experiment resume \
                -c "$CONTAINER_PROJECT_ROOT/configs/experiment_910c_kimi_k3.yaml" \
                --task "$TASK_ID" --allow-missing-baseline
        else
            set --
        fi
        docker run --detach \
            --name "$CONTROLLER_NAME" \
            --runtime=runc \
            --user "$CONTROLLER_UID:$SHARED_GID" \
            --restart on-failure:3 \
            --read-only \
            --cap-drop ALL \
            --security-opt no-new-privileges \
            --pids-limit 512 \
            --memory "${AKG_CONTROLLER_MEMORY:-8g}" \
            --log-opt "max-size=${AKG_LOG_MAX_SIZE:-20m}" \
            --log-opt "max-file=${AKG_LOG_MAX_FILES:-5}" \
            --tmpfs /tmp:rw,nosuid,nodev,noexec,mode=1777 \
            --env-file "$CONTROLLER_ENV_FILE" \
            --env-file "$HIDDEN_ENV_FILE" \
            --env "AKG_PROJECT_ROOT=$CONTAINER_PROJECT_ROOT" \
            --env "AKG_CONFIG_PATH=$CONTAINER_PROJECT_ROOT/configs/experiment_910c_kimi_k3.yaml" \
            --env HOME=/tmp/akg-home \
            --env GIT_CONFIG_COUNT=1 \
            --env GIT_CONFIG_KEY_0=safe.directory \
            --env "GIT_CONFIG_VALUE_0=$CONTAINER_PROJECT_ROOT" \
            --env GIT_OPTIONAL_LOCKS=0 \
            --volume "$PROJECT_ROOT:$CONTAINER_PROJECT_ROOT:ro" \
            --volume "$PROJECT_ROOT/runs:$CONTAINER_PROJECT_ROOT/runs:rw" \
            "$CONTROLLER_IMAGE" "$@"
        ;;
esac
