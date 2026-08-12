#!/bin/sh
set -eu

usage() {
    echo "usage: $0 SSH_HOST GIT_URL ABSOLUTE_REMOTE_DIR CANN_SET_ENV [GIT_REF]" >&2
}

if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
    usage
    exit 2
fi

SSH_HOST=$1
GIT_URL=$2
REMOTE_DIR=$3
CANN_SET_ENV=$4
GIT_REF=${5:-}

case "$SSH_HOST" in
    ""|-*)
        echo "error: invalid SSH host" >&2
        exit 2
        ;;
esac

case "$REMOTE_DIR" in
    /|"")
        echo "error: refusing a broad or empty remote directory" >&2
        exit 2
        ;;
    /*) ;;
    *)
        echo "error: remote directory must be absolute" >&2
        exit 2
        ;;
esac

case "$CANN_SET_ENV" in
    /*) ;;
    *)
        echo "error: CANN set_env path must be absolute on the remote host" >&2
        exit 2
        ;;
esac

# Authentication is delegated to the caller's existing SSH/Git setup. This
# script never copies API tokens or SSH private keys to the remote host.
ssh -- "$SSH_HOST" sh -s -- "$GIT_URL" "$REMOTE_DIR" "$CANN_SET_ENV" "$GIT_REF" <<'REMOTE'
set -eu

repo_url=$1
destination=$2
cann_env=$3
git_ref=$4

case "$repo_url" in
    "")
        echo "error: repository URL must not be empty" >&2
        exit 2
        ;;
esac

case "$destination" in
    /|"")
        echo "error: refusing broad destination" >&2
        exit 2
        ;;
    /*) ;;
    *)
        echo "error: destination must be absolute" >&2
        exit 2
        ;;
esac

if [ ! -r "$cann_env" ]; then
    echo "error: CANN environment file is not readable: $cann_env" >&2
    exit 2
fi
if ! command -v git >/dev/null 2>&1; then
    echo "error: git is not installed on the remote host" >&2
    exit 2
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "error: Python 3 is not installed on the remote host" >&2
    exit 2
fi
if ! command -v getent >/dev/null 2>&1 || ! getent group ascend-kernel >/dev/null; then
    echo "error: pre-provision the ascend-kernel deployment group" >&2
    exit 2
fi
if [ "$(id -u)" -ne 0 ]; then
    if ! command -v sudo >/dev/null 2>&1 || ! sudo -n true; then
        echo "error: remote installer requires root or non-interactive sudo" >&2
        exit 2
    fi
fi

if [ -e "$destination" ]; then
    echo "error: destination already exists; refusing to overwrite: $destination" >&2
    exit 3
fi

if [ -n "$git_ref" ]; then
    git clone --branch "$git_ref" --single-branch -- "$repo_url" "$destination"
else
    git clone -- "$repo_url" "$destination"
fi

cd "$destination"
head=$(git rev-parse --verify HEAD)
if [ "${#head}" -ne 40 ]; then
    echo "error: cloned HEAD is not a 40-character SHA-1" >&2
    exit 3
fi
case "$head" in
    *[!0-9a-f]*)
        echo "error: cloned HEAD is not a lowercase SHA-1" >&2
        exit 3
        ;;
esac
if [ -n "$(git status --porcelain=v1 --untracked-files=all)" ]; then
    echo "error: clone is dirty before installation" >&2
    exit 3
fi
if [ "$(id -u)" -eq 0 ]; then
    AKG_CANN_ENV_FILE=$cann_env \
    AKG_VENV_DIR=${destination}-venv \
    ./scripts/install-910c.sh
else
    sudo -n env \
        AKG_CANN_ENV_FILE=$cann_env \
        AKG_VENV_DIR=${destination}-venv \
        AKG_SHARED_GROUP=ascend-kernel \
        AKG_VENV_OWNER=root \
        ./scripts/install-910c.sh
fi

if [ "$(git rev-parse --verify HEAD)" != "$head" ] || \
   [ -n "$(git status --porcelain=v1 --untracked-files=all)" ]; then
    echo "error: clone changed during installation" >&2
    exit 4
fi
printf '%s\n' "$head"
echo "Remote clone and installation completed at $destination"
REMOTE
