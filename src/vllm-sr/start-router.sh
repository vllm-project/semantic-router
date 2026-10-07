#!/bin/bash
# Start script for router service
# Starts the router directly from canonical config.yaml

set -e

# The image's entrypoint for every launcher. With no arguments, or with Router
# flags (the Helm chart, the Operator, manifests written for the former extproc
# image), it runs the Router on the Kubernetes config mount, flags appended.
# With a config path, it starts the Router of a `vllm-sr serve` stack.
if [ $# -eq 0 ] || [ "${1#-}" != "$1" ]; then
    exec /usr/local/bin/router --config=/app/config/config.yaml "$@"
fi

CONFIG_FILE="${1:-/app/config.yaml}"
echo "Starting router from canonical config..."
echo "  Config file: $CONFIG_FILE"

# Mark wildcard management listeners as container-internal. The listener's
# actual bind address and port remain authoritative in canonical config; host
# publication is independently constrained by the split-stack launcher.
export VLLM_SR_MANAGEMENT_INTERNAL_LISTENER=true

# Preserve setup-mode behavior from the historical single-container entrypoint.
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

# VLLM_SR_GATEWAY=standalone makes the Router serve the listeners itself. They
# bind every container address; the listener's configured address governs only
# the host publication. Any other value serves ext_proc, for Envoy in front.
GATEWAY_ARGS=()
if [ "${VLLM_SR_GATEWAY:-}" = "standalone" ]; then
    GATEWAY_ARGS=(-gateway=standalone -listener-address=0.0.0.0)
fi

# Start router
echo "Starting router (gateway: ${VLLM_SR_GATEWAY:-extproc})..."
exec /usr/local/bin/router \
    -config="$CONFIG_FILE" \
    -port=50051 \
    -enable-api=true \
    "${GATEWAY_ARGS[@]}"
