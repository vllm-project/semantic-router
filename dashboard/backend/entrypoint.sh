#!/bin/sh
set -eu

# Fix ownership and permissions for config file before switching to nonroot user
# This allows the dashboard to write to the mounted config.yaml file
CONFIG_FILE_PATH=${ROUTER_CONFIG_PATH:-/app/config/config.yaml}
STATE_DIR=$(dirname "$CONFIG_FILE_PATH")
RECIPE_STORE_DIR=${VLLM_SR_RECIPE_STORE_DIR:-${STATE_DIR}/.vllm-sr/recipe-store}
PERMISSION_HELPER=/app/entrypoint_permissions.py
SERVER_READONLY=${DASHBOARD_READONLY:-false}
RUNTIME_CONFIG_WRITABLE=${DASHBOARD_RUNTIME_CONFIG_WRITABLE:-true}
RECIPE_STORE_WRITABLE=${DASHBOARD_RECIPE_STORE_WRITABLE:-true}
LOG_SPOOL_GID=${VLLM_SR_LOG_SPOOL_GID:-}

# OpenShift restricted SCCs run images with an arbitrary non-root UID that is
# a member of the root group. Such a process cannot prepare users or bind
# mounts, so keep runtime mutations fail-closed and start directly. The image
# grants group 0 write access only to /app/data for Dashboard-owned state.
if [ "$(id -u)" -ne 0 ]; then
    DASHBOARD_RUNTIME_CONFIG_WRITABLE=false
    DASHBOARD_RECIPE_STORE_WRITABLE=false
    OPENCLAW_CONTAINER_RUNTIME_DISABLED=true
    export DASHBOARD_RUNTIME_CONFIG_WRITABLE DASHBOARD_RECIPE_STORE_WRITABLE \
        OPENCLAW_CONTAINER_RUNTIME_DISABLED
    exec "$@"
fi

add_nonroot_group_gid() {
    GROUP_GID=$1
    if [ "$GROUP_GID" -eq 0 ]; then
        echo "Refusing to add Dashboard user to the root group" >&2
        exit 1
    fi
    GROUP_NAME=$(getent group "$GROUP_GID" | cut -d: -f1 || true)
    if [ -z "$GROUP_NAME" ]; then
        GROUP_NAME="dashboard-share-${GROUP_GID}"
        groupadd --gid "$GROUP_GID" "$GROUP_NAME"
    fi
    usermod -aG "$GROUP_NAME" nonroot
}

# The local split runtime exposes its bounded log spool through a private host
# group. The backend receives read membership only; the reader mount itself is
# read-only and producer output is captured by the outer PID 1 relay.
if [ -n "$LOG_SPOOL_GID" ]; then
    case "$LOG_SPOOL_GID" in
        *[!0-9]*|0) echo "Invalid log spool group" >&2; exit 1 ;;
    esac
    add_nonroot_group_gid "$LOG_SPOOL_GID"
fi

# Atomic runtime-config replacement needs group write access to the containing
# state directory, not only to the current file. Preserve the host owner's UID
# so the host CLI and the nonroot Dashboard can both manage the state. A
# declared read-only Dashboard may consume a read-only ConfigMap without these
# mutations. Config mount availability is intentionally independent from the
# managed Recipe package store: an admin may still import packages for later
# activation when only the runtime config mount is read-only.
if [ "$SERVER_READONLY" != "true" ] && [ "$RUNTIME_CONFIG_WRITABLE" = "true" ]; then
    if [ ! -f "$CONFIG_FILE_PATH" ] || [ ! -d "$STATE_DIR" ] ||
       ! python3 "$PERMISSION_HELPER" probe-config "$STATE_DIR" "$CONFIG_FILE_PATH"; then
        echo "Dashboard runtime config is read-only; runtime mutation is disabled while Recipe package storage remains independent" >&2
        RUNTIME_CONFIG_WRITABLE=false
    fi
fi

if [ "$SERVER_READONLY" != "true" ] && [ "$RECIPE_STORE_WRITABLE" = "true" ]; then
    if ! python3 "$PERMISSION_HELPER" ensure-directory "$RECIPE_STORE_DIR"; then
        echo "Dashboard Recipe package store is read-only; package import is disabled" >&2
        RECIPE_STORE_WRITABLE=false
    fi
fi

DASHBOARD_RUNTIME_CONFIG_WRITABLE=$RUNTIME_CONFIG_WRITABLE
DASHBOARD_RECIPE_STORE_WRITABLE=$RECIPE_STORE_WRITABLE
export DASHBOARD_RUNTIME_CONFIG_WRITABLE DASHBOARD_RECIPE_STORE_WRITABLE

if [ "$SERVER_READONLY" != "true" ] && [ "$RUNTIME_CONFIG_WRITABLE" = "true" ]; then
    STATE_GID=65532
    if [ -d "$STATE_DIR" ]; then
        add_nonroot_group_gid "$STATE_GID"
        python3 "$PERMISSION_HELPER" prepare-directory "$STATE_DIR" "$STATE_GID"
    fi
    if [ -f "$CONFIG_FILE_PATH" ]; then
        python3 "$PERMISSION_HELPER" prepare-file "$CONFIG_FILE_PATH" "$STATE_GID"
    fi
    if [ -n "${VLLM_SR_ENVOY_CONFIG_PATH:-}" ] && [ -f "$VLLM_SR_ENVOY_CONFIG_PATH" ]; then
        ENVOY_STATE_DIR=$(dirname "$VLLM_SR_ENVOY_CONFIG_PATH")
        ENVOY_STATE_GID=65532
        add_nonroot_group_gid "$ENVOY_STATE_GID"
        python3 "$PERMISSION_HELPER" prepare-directory "$ENVOY_STATE_DIR" "$ENVOY_STATE_GID"
        python3 "$PERMISSION_HELPER" prepare-file "$VLLM_SR_ENVOY_CONFIG_PATH" "$ENVOY_STATE_GID"
    fi
fi
if [ "$SERVER_READONLY" != "true" ] && [ "$RECIPE_STORE_WRITABLE" = "true" ] &&
   [ -d "$RECIPE_STORE_DIR" ]; then
    RECIPE_STORE_GID=65532
    add_nonroot_group_gid "$RECIPE_STORE_GID"
    python3 "$PERMISSION_HELPER" prepare-tree \
        "$RECIPE_STORE_DIR" "$RECIPE_STORE_GID" \
        --credential-relative-path credentials/router-management.token
fi
if [ -d /app/data ]; then
    DATA_GID=65532
    add_nonroot_group_gid "$DATA_GID"
    python3 "$PERMISSION_HELPER" prepare-tree /app/data "$DATA_GID"
fi

# The dashboard is deliberately nonroot, but managed Recipe topology and
# OpenClaw operations use the mounted container-runtime socket. Map the
# socket's numeric group inside the image before gosu rebuilds supplementary
# groups for the nonroot account. Never broaden the socket's host permissions.
CONTAINER_SOCKET_PATH=${VLLM_SR_CONTAINER_SOCKET_PATH:-/var/run/docker.sock}
if [ -e "$CONTAINER_SOCKET_PATH" ] || [ -L "$CONTAINER_SOCKET_PATH" ]; then
    if CONTAINER_SOCKET_GID=$(python3 "$PERMISSION_HELPER" socket-gid "$CONTAINER_SOCKET_PATH" 2>/dev/null); then
        add_nonroot_group_gid "$CONTAINER_SOCKET_GID"
        export OPENCLAW_CONTAINER_RUNTIME_DISABLED=false
    else
        export OPENCLAW_CONTAINER_RUNTIME_DISABLED=true
        echo "Warning: Dashboard container management is unavailable because the runtime socket cannot be shared safely; continuing without socket access" >&2
    fi
else
    export OPENCLAW_CONTAINER_RUNTIME_DISABLED=true
fi

# Switch to nonroot user and execute the dashboard backend.
if ! command -v gosu >/dev/null 2>&1; then
    echo "gosu is required to initialize Dashboard supplementary groups" >&2
    exit 1
fi
exec gosu nonroot "$@"
