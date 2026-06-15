#!/bin/bash
set -e

IMAGE="ghcr.io/vllm-project/semantic-router/vllm-sr:latest"
ROUTER_IMAGE="${IMAGE}"
ENVOY_IMAGE="envoyproxy/envoy:v1.34-latest"
DASHBOARD_IMAGE="ghcr.io/vllm-project/semantic-router/dashboard:latest"
RUNTIME_CONTAINERS=(
  vllm-sr-router-container
  vllm-sr-envoy-container
  vllm-sr-dashboard-container
  vllm-sr-sim-container
  vllm-sr-prometheus
  vllm-sr-grafana
  vllm-sr-jaeger
)

echo "=========================================="
echo "Rebuild and Test vLLM Semantic Router"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Clean up old containers"
echo "  2. Rebuild Docker images with all dependencies"
echo "  3. Start the service"
echo "  4. Verify multi-listener support"
echo ""
echo "Router image checks:"
echo "  ✓ router binary"
echo "  ✓ setup-mode YAML helper"
echo "  ✓ router health probe tooling"
echo ""
read -r -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Clean up old containers
echo "1. Cleaning up old containers..."
docker rm -f "${RUNTIME_CONTAINERS[@]}" 2>/dev/null || echo "  No containers to remove"
echo ""

# Rebuild Docker images
echo "2. Rebuilding Docker images..."
echo "  Building from: $(pwd)/../.."
echo "  Note: Use 'make docker-buildx' for multi-platform builds"
echo ""
cd ../..
docker build -t "${IMAGE}" -f src/vllm-sr/Dockerfile .
docker image inspect "${ENVOY_IMAGE}" >/dev/null 2>&1 || docker pull "${ENVOY_IMAGE}"
docker build -t "${DASHBOARD_IMAGE}" -f dashboard/backend/Dockerfile .
cd src/vllm-sr
echo ""
echo "✓ Router image built: ${IMAGE}"
echo "✓ Official Envoy image available: ${ENVOY_IMAGE}"
echo "✓ Dashboard image built: ${DASHBOARD_IMAGE}"
echo ""

# Verify dependencies in images
echo "3. Verifying dependencies in images..."
docker run --rm --entrypoint /bin/sh "${ROUTER_IMAGE}" -c "
    echo '  Checking router binary...'
    test -x /usr/local/bin/router && echo '    ✓ router binary present'
    echo '  Checking setup helper yaml import...'
    python3 -c 'import yaml; print(\"    ✓ yaml import ok\")'
    echo '  Checking curl...'
    curl --version | head -1 | sed 's/^/    ✓ /'
"
docker run --rm --entrypoint /bin/sh "${ENVOY_IMAGE}" -c "
    echo '  Checking Envoy...'
    envoy --version | head -1 | sed 's/^/    ✓ /'
"
docker run --rm --entrypoint /bin/sh "${DASHBOARD_IMAGE}" -c "
    echo '  Checking dashboard backend...'
    /usr/local/bin/dashboard-backend -help >/dev/null && echo '    ✓ dashboard backend present'
    echo '  Checking auth bootstrap endpoint...'
    mkdir -p /tmp/static /tmp/data
    printf '<!doctype html><title>ok</title>' > /tmp/static/index.html
    printf 'version: v0.3\nlisteners:\n  - name: http\n    address: 0.0.0.0\n    port: 8899\n' > /tmp/config.yaml
    /usr/local/bin/dashboard-backend -port=8700 -static=/tmp/static -config=/tmp/config.yaml -auth-db /tmp/data/auth.db -auth-jwt-secret test-secret >/tmp/dashboard.log 2>&1 &
    pid=\$!
    for i in 1 2 3 4 5; do
        if curl -fsS http://127.0.0.1:8700/api/auth/bootstrap/can-register >/tmp/auth.json 2>/dev/null; then break; fi
        sleep 1
    done
    grep -q '\"canRegister\":true' /tmp/auth.json && echo '    ✓ auth bootstrap endpoint ok' || { cat /tmp/dashboard.log; exit 1; }
    kill \$pid >/dev/null 2>&1 || true
    wait \$pid >/dev/null 2>&1 || true
    echo '  Checking Docker CLI...'
    docker --version | sed 's/^/    ✓ /'
    echo '  Checking runtime helper import...'
    python3 -c 'import cli.commands.runtime_support; print(\"    ✓ runtime helper import ok\")'
"
echo ""

# Show config listeners
echo "4. Checking config.yaml listeners..."
python -c "
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    listeners = config.get('listeners', [])
    print(f'  Found {len(listeners)} listener(s):')
    for listener in listeners:
        name = listener.get('name', 'unknown')
        port = listener.get('port', 'unknown')
        print(f'    - {name}: port {port}')
"
echo ""

# Start service
echo "5. Starting service with local images..."
VLLM_SR_IMAGE="${IMAGE}" VLLM_SR_ROUTER_IMAGE="${ROUTER_IMAGE}" VLLM_SR_ENVOY_IMAGE="${ENVOY_IMAGE}" VLLM_SR_DASHBOARD_IMAGE="${DASHBOARD_IMAGE}" python -m cli.main serve --config config.yaml --image-pull-policy never
echo ""

# Wait a bit for startup
echo "6. Waiting for services to start (30 seconds)..."
sleep 30
echo ""

# Check service status
echo "7. Service status:"
python -m cli.main status || true
echo ""

# Show managed containers
echo "8. Managed containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep 'vllm-sr-' || true
echo ""

# Show logs
echo "9. Recent router logs:"
python -m cli.main logs router || true
echo ""

echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "Check health:"
echo "  curl http://localhost:8000/healthz"
echo "  curl http://localhost:8080/healthz"
echo ""
echo "View logs:"
echo "  python -m cli.main logs router --follow"
echo ""
echo "Open shell:"
echo "  docker exec -it vllm-sr-router-container /bin/bash"
echo ""
echo "Stop service:"
echo "  vllm-sr stop"
echo ""
