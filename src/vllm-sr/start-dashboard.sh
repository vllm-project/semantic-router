#!/bin/sh
# Start dashboard with dynamically determined Envoy port from config.yaml

CONFIG_FILE="${1:-/app/config.yaml}"

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

echo "Starting dashboard with Envoy at http://localhost:${ENVOY_PORT}"

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
    -config=/app/config.yaml \
    -router_api=http://localhost:8080 \
    -router_metrics=http://localhost:9190/metrics \
    -envoy="http://localhost:${ENVOY_PORT}" \
    ${READONLY_ARG} \
    "${OBSERVABILITY_ARGS}"

