#!/bin/sh
set -eu

PERMISSION_HELPER=/app/entrypoint_permissions.py
LOG_SPOOL_GID=${VLLM_SR_LOG_SPOOL_GID:-}

# OpenShift restricted SCCs run images with an arbitrary non-root UID that is
# a member of the root group. Such a process cannot prepare users or bind
# mounts, so keep container-runtime operations fail-closed and start directly. The image
# grants group 0 write access only to /app/data for Dashboard-owned state.
if [ "$(id -u)" -ne 0 ]; then
    OPENCLAW_CONTAINER_RUNTIME_DISABLED=true
    export OPENCLAW_CONTAINER_RUNTIME_DISABLED
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

if [ -d /app/data ]; then
    DATA_GID=65532
    add_nonroot_group_gid "$DATA_GID"
    python3 "$PERMISSION_HELPER" prepare-tree /app/data "$DATA_GID"
fi

# The first-registration saga alone may consume this one-time credential. Its
# dedicated bind mount contains no long-lived Router or Dashboard key material.
if [ -n "${DASHBOARD_ROUTER_BOOTSTRAP_TOKEN_FILE:-}" ]; then
    python3 "$PERMISSION_HELPER" prepare-bootstrap-token \
        "$DASHBOARD_ROUTER_BOOTSTRAP_TOKEN_FILE" 65532 65532
fi

# Long-lived Dashboard issuer material is copied into an isolated ephemeral
# runtime directory. The non-root backend never receives traversal access to
# the Router state tree that contains unrelated control-plane secrets.
STAGED_SECRET_DIR=/run/vllm-sr-dashboard-secrets
mkdir -p "$STAGED_SECRET_DIR"
chown root:root "$STAGED_SECRET_DIR"
chmod 0700 "$STAGED_SECRET_DIR"
if [ -n "${DASHBOARD_SIGNING_KEY_FILE:-}" ]; then
    python3 "$PERMISSION_HELPER" stage-private-file \
        "$DASHBOARD_SIGNING_KEY_FILE" "$STAGED_SECRET_DIR/signing-key.pem" 65532 65532
    DASHBOARD_SIGNING_KEY_FILE="$STAGED_SECRET_DIR/signing-key.pem"
    export DASHBOARD_SIGNING_KEY_FILE
fi
if [ -n "${DASHBOARD_ISSUER_TLS_CERT_FILE:-}" ]; then
    python3 "$PERMISSION_HELPER" stage-private-file \
        "$DASHBOARD_ISSUER_TLS_CERT_FILE" "$STAGED_SECRET_DIR/tls-cert.pem" 65532 65532
    DASHBOARD_ISSUER_TLS_CERT_FILE="$STAGED_SECRET_DIR/tls-cert.pem"
    export DASHBOARD_ISSUER_TLS_CERT_FILE
fi
if [ -n "${DASHBOARD_ISSUER_TLS_KEY_FILE:-}" ]; then
    python3 "$PERMISSION_HELPER" stage-private-file \
        "$DASHBOARD_ISSUER_TLS_KEY_FILE" "$STAGED_SECRET_DIR/tls-key.pem" 65532 65532
    DASHBOARD_ISSUER_TLS_KEY_FILE="$STAGED_SECRET_DIR/tls-key.pem"
    export DASHBOARD_ISSUER_TLS_KEY_FILE
fi
if [ -n "${SSL_CERT_FILE:-}" ]; then
    python3 "$PERMISSION_HELPER" stage-private-file \
        "$SSL_CERT_FILE" "$STAGED_SECRET_DIR/trust-bundle.pem" 65532 65532
    SSL_CERT_FILE="$STAGED_SECRET_DIR/trust-bundle.pem"
    export SSL_CERT_FILE
fi
chown 65532:65532 "$STAGED_SECRET_DIR"
chmod 0700 "$STAGED_SECRET_DIR"

# The dashboard is deliberately nonroot, but OpenClaw operations use the
# mounted container-runtime socket. Map the
# socket's numeric group inside the image before gosu rebuilds supplementary
# groups for the nonroot account. The local CLI sets the flag to false only
# after an explicit VLLM_SR_CONTAINER_SOCKET opt-in and host-side validation.
# Never broaden the socket's host permissions.
CONTAINER_SOCKET_PATH=${VLLM_SR_CONTAINER_SOCKET_PATH:-/var/run/docker.sock}
if [ "${OPENCLAW_CONTAINER_RUNTIME_DISABLED:-true}" != "false" ]; then
    export OPENCLAW_CONTAINER_RUNTIME_DISABLED=true
elif [ -e "$CONTAINER_SOCKET_PATH" ] || [ -L "$CONTAINER_SOCKET_PATH" ]; then
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
