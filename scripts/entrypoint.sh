#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE_PATH=${CONFIG_FILE:-/app/config/config.yaml}
AI_BINDING=${AI_BINDING:-candle}

if [[ ! -f "$CONFIG_FILE_PATH" ]]; then
  echo "[entrypoint] Config file not found at $CONFIG_FILE_PATH" >&2
  exit 1
fi

case "$AI_BINDING" in
  onnx)
    BINARY=/app/router-onnx
    ;;
  candle|"")
    BINARY=/app/router-candle
    ;;
  *)
    echo "[entrypoint] Unknown AI_BINDING='$AI_BINDING'. Valid values: candle (default), onnx" >&2
    exit 1
    ;;
esac

if [[ ! -f "$BINARY" ]]; then
  echo "[entrypoint] Binary not found: $BINARY (AI_BINDING=$AI_BINDING)" >&2
  echo "[entrypoint] Falling back to candle binding..." >&2
  BINARY=/app/router-candle
  AI_BINDING=candle
  if [[ ! -f "$BINARY" ]]; then
    echo "[entrypoint] Fallback binary also not found: $BINARY" >&2
    exit 1
  fi
fi

echo "[entrypoint] Starting semantic-router with AI_BINDING=$AI_BINDING"
echo "[entrypoint] Config: $CONFIG_FILE_PATH"
echo "[entrypoint] Additional args: $*"
exec "$BINARY" --config "$CONFIG_FILE_PATH" "$@"
