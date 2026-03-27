#!/bin/sh
# Start dashboard with dynamically determined Envoy port from config.yaml

CONFIG_FILE="${1:-/app/config.yaml}"

# Check if dashboard is disabled (minimal mode)
if [ "${DISABLE_DASHBOARD}" = "true" ]; then
    echo "Dashboard disabled (minimal mode). Sleeping..."
    exec sleep infinity
fi

# Extract the first listener port from config.yaml using Python
ENVOY_PORT=$(python3 -c "
import yaml
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    listeners = config.get('listeners', [])
    if listeners:
        port = listeners[0].get('port', 8888)
        print(port)
    else:
        print(8888)
except Exception as e:
    print(8888)
")

ROUTER_API_URL="${TARGET_ROUTER_API_URL:-http://localhost:8080}"
ROUTER_METRICS_URL="${TARGET_ROUTER_METRICS_URL:-http://localhost:9190/metrics}"
ENVOY_URL="${TARGET_ENVOY_URL:-http://localhost:${ENVOY_PORT}}"

echo "Starting dashboard with Router API at ${ROUTER_API_URL}"
echo "Starting dashboard with Router metrics at ${ROUTER_METRICS_URL}"
echo "Starting dashboard with Envoy at ${ENVOY_URL}"

# Check for read-only mode
READONLY_ARG=""
if [ "${DASHBOARD_READONLY}" = "true" ]; then
    READONLY_ARG="-readonly"
    echo "Dashboard read-only mode: ENABLED"
fi

# Build observability arguments
OBSERVABILITY_ARGS=""
if [ -n "${TARGET_JAEGER_URL}" ]; then
    OBSERVABILITY_ARGS="${OBSERVABILITY_ARGS} -jaeger=${TARGET_JAEGER_URL}"
    echo "Jaeger URL: ${TARGET_JAEGER_URL}"
fi
if [ -n "${TARGET_GRAFANA_URL}" ]; then
    OBSERVABILITY_ARGS="${OBSERVABILITY_ARGS} -grafana=${TARGET_GRAFANA_URL}"
    echo "Grafana URL: ${TARGET_GRAFANA_URL}"
fi
if [ -n "${TARGET_PROMETHEUS_URL}" ]; then
    OBSERVABILITY_ARGS="${OBSERVABILITY_ARGS} -prometheus=${TARGET_PROMETHEUS_URL}"
    echo "Prometheus URL: ${TARGET_PROMETHEUS_URL}"
fi

exec /usr/local/bin/dashboard-backend \
    -port=8700 \
    -static=/app/frontend \
    -config="$CONFIG_FILE" \
    -router_api="${ROUTER_API_URL}" \
    -router_metrics="${ROUTER_METRICS_URL}" \
    -envoy="${ENVOY_URL}" \
    ${READONLY_ARG} \
    "${OBSERVABILITY_ARGS}"
