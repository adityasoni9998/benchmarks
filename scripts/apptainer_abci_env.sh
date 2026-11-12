#!/usr/bin/env bash
# Source this file inside an ABCI allocation before pulling/building SIFs.

apptainer_abci_env() {
    local scratch account
    scratch="${ABCI_SCRATCH:-${PBS_LOCALDIR:-}}"
    if [[ -z "$scratch" || ! -d "$scratch" || ! -w "$scratch" ]]; then
        echo 'Set ABCI_SCRATCH (or PBS_LOCALDIR) to your writable allocation directory under /local.' >&2
        return 1
    fi
    scratch="$(realpath -- "$scratch")" || return 1
    case "$scratch" in
        /local/*) ;;
        *) echo "Expected allocation-local scratch under /local, got: $scratch" >&2; return 1 ;;
    esac

    account="${USER_ID:-$(id -un)}"
    if [[ ! "$account" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        echo 'USER_ID must be an account name without path separators.' >&2
        return 1
    fi
    umask 077
    export CONTAINER_ROOT="/groups/gai51740/$account/apptainer"
    export OPENHANDS_APPTAINER_BUILD_ROOT="$CONTAINER_ROOT/sif"
    export APPTAINER_CACHEDIR="$CONTAINER_ROOT/cache/apptainer"
    export APPTAINER_TMPDIR="$scratch/apptainer"
    export TMPDIR="$APPTAINER_TMPDIR"
    export UV_CACHE_DIR="$CONTAINER_ROOT/cache/uv"
    export HF_HOME="$CONTAINER_ROOT/cache/huggingface"
    mkdir -p "$OPENHANDS_APPTAINER_BUILD_ROOT" "$APPTAINER_CACHEDIR" \
        "$APPTAINER_TMPDIR" "$UV_CACHE_DIR" "$HF_HOME" \
        "$CONTAINER_ROOT/manifests" "$CONTAINER_ROOT/logs" || return 1
    chmod 700 "$CONTAINER_ROOT" "$APPTAINER_CACHEDIR" \
        "$APPTAINER_TMPDIR" || return 1
}

apptainer_abci_env
