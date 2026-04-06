#!/bin/bash
# Start script for router service
# Starts the router directly from canonical config.yaml

set -e

CONFIG_FILE="${VLLM_SR_RUNTIME_CONFIG_PATH:-/app/config.yaml}"
if [ "$#" -gt 0 ] && [[ "$1" != -* ]]; then
    CONFIG_FILE="$1"
    shift
fi
echo "Starting router from canonical config..."
echo "  Config file: $CONFIG_FILE"

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
    echo "Setup mode enabled: router disabled"
    exec sleep infinity
fi

# Start router
echo "Starting router..."
exec /usr/local/bin/router \
    -config="$CONFIG_FILE" \
    -port=50051 \
    -enable-api=true \
    -api-port=8080 \
    "$@"
