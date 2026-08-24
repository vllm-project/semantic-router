#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE_PATH=${CONFIG_FILE:-/app/config/config.yaml}
AI_BINDING=${AI_BINDING:-candle}

merge_custom_ca_with_system_roots() {
  local custom_bundle="${SSL_CERT_FILE:-}"
  local system_bundle=""
  local candidate
  local combined_bundle

  if [[ -z "$custom_bundle" ]]; then
    return
  fi
  if [[ ! -f "$custom_bundle" || ! -r "$custom_bundle" ]]; then
    echo "[entrypoint] Configured SSL_CERT_FILE is not a readable file" >&2
    exit 1
  fi

  for candidate in \
    /etc/ssl/certs/ca-certificates.crt \
    /etc/pki/tls/certs/ca-bundle.crt \
    /etc/ssl/ca-bundle.pem; do
    if [[ ! -f "$candidate" || ! -r "$candidate" ]]; then
      continue
    fi
    if [[ "$custom_bundle" -ef "$candidate" ]]; then
      return
    fi
    system_bundle="$candidate"
    break
  done
  if [[ -z "$system_bundle" ]]; then
    echo "[entrypoint] Cannot preserve public TLS trust: no readable system CA bundle was found" >&2
    exit 1
  fi

  combined_bundle=$(mktemp /tmp/vllm-sr-ca-bundle.XXXXXX)
  {
    cat "$system_bundle"
    printf '\n'
    cat "$custom_bundle"
    printf '\n'
  } > "$combined_bundle"
  chmod 0600 "$combined_bundle"
  SSL_CERT_FILE="$combined_bundle"
  export SSL_CERT_FILE
}

if [[ ! -f "$CONFIG_FILE_PATH" ]]; then
  echo "[entrypoint] Config file not found at $CONFIG_FILE_PATH" >&2
  exit 1
fi

case "$AI_BINDING" in
  onnx)
    BINARY=/app/router-onnx
    ;;
  openvino)
    BINARY=/app/router-openvino
    ;;
  candle|"")
    BINARY=/app/router-candle
    ;;
  *)
    echo "[entrypoint] Unknown AI_BINDING='$AI_BINDING'. Valid values: candle (default), onnx, openvino" >&2
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

merge_custom_ca_with_system_roots

echo "[entrypoint] Starting semantic-router with AI_BINDING=$AI_BINDING"
echo "[entrypoint] Config: $CONFIG_FILE_PATH"
echo "[entrypoint] Additional args: $*"
exec "$BINARY" --config "$CONFIG_FILE_PATH" "$@"
