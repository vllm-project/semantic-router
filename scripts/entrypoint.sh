#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE_PATH=${CONFIG_FILE:-/app/config/config.yaml}

if [[ ! -f "$CONFIG_FILE_PATH" ]]; then
  echo "[entrypoint] Config file not found at $CONFIG_FILE_PATH" >&2
  exit 1
fi

echo "[entrypoint] Starting semantic-router with config: $CONFIG_FILE_PATH"
echo "[entrypoint] Additional args: $*"
exec /app/extproc-server --config "$CONFIG_FILE_PATH" "$@"
