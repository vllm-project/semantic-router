#!/bin/sh
set -eu

# Fix ownership and permissions for config file before switching to nonroot user
# This allows the dashboard to write to the mounted config.yaml file
CONFIG_FILE_PATH=${ROUTER_CONFIG_PATH:-/app/config/config.yaml}

if [ -f "$CONFIG_FILE_PATH" ]; then
    # Change ownership to nonroot user (UID 65532) so they can write to it
    # Use chown with || true to avoid failing if we can't change ownership (e.g., on some mount types)
    chown nonroot:nonroot "$CONFIG_FILE_PATH" 2>/dev/null || {
        # Fallback: make it group-writable (664) - requires nonroot to be in the file's group
        chmod 664 "$CONFIG_FILE_PATH" 2>/dev/null || true
    }
fi

# Switch to nonroot user and execute the dashboard backend.
if command -v gosu >/dev/null 2>&1; then
    exec gosu nonroot:nonroot "$@"
fi

if command -v su-exec >/dev/null 2>&1; then
    exec su-exec nonroot:nonroot "$@"
fi

echo "Neither gosu nor su-exec is available" >&2
exit 1
