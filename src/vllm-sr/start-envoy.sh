#!/bin/sh
# Start script for standalone Envoy service container.

set -e

CONFIG_FILE="${1:-${VLLM_SR_RUNTIME_CONFIG_PATH:-/app/config.yaml}}"
ENVOY_CONFIG_FILE="${2:-/etc/envoy/envoy.yaml}"

echo "Starting Envoy..."
echo "  Config file: ${CONFIG_FILE}"
echo "  Envoy config: ${ENVOY_CONFIG_FILE}"

# Preserve setup-mode behavior from the legacy supervisord entrypoint.
if python3 -c "
import sys, yaml
try:
    data = yaml.safe_load(open('$CONFIG_FILE')) or {}
    setup = data.get('setup')
    sys.exit(0 if isinstance(setup, dict) and setup.get('mode') else 1)
except Exception:
    sys.exit(1)
"; then
    echo "Setup mode enabled: envoy disabled"
    exec sleep infinity
fi

python3 -m cli.config_generator "${CONFIG_FILE}" "${ENVOY_CONFIG_FILE}"
exec /usr/local/bin/envoy \
    -c "${ENVOY_CONFIG_FILE}" \
    --log-level debug \
    --log-format '[%Y-%m-%d %T.%e][%l] %v'
