#!/bin/bash
# Entrypoint Script for vLLM Semantic Router Stack

set -e

# Default environment variables
export CONFIG_FILE="${CONFIG_FILE:-/app/config/config.yaml}"
export ENVOY_LISTEN_PORT="${ENVOY_LISTEN_PORT:-8801}"
export ENVOY_ADMIN_PORT="${ENVOY_ADMIN_PORT:-19000}"
export EXTPROC_HOST="${EXTPROC_HOST:-localhost}"
export EXTPROC_PORT="${EXTPROC_PORT:-50051}"
export DASHBOARD_PORT="${DASHBOARD_PORT:-8700}"
export GF_SECURITY_ADMIN_USER="${GF_SECURITY_ADMIN_USER:-admin}"
export GF_SECURITY_ADMIN_PASSWORD="${GF_SECURITY_ADMIN_PASSWORD:-admin}"

# Dashboard service URLs (for stack deployment, services run on localhost)
export TARGET_GRAFANA_URL="${TARGET_GRAFANA_URL:-http://localhost:3000}"
export TARGET_JAEGER_URL="${TARGET_JAEGER_URL:-http://localhost:16686}"
export TARGET_PROMETHEUS_URL="${TARGET_PROMETHEUS_URL:-http://localhost:9090}"
export TARGET_CHATUI_URL="${TARGET_CHATUI_URL:-http://localhost:${CHATUI_PORT:-5173}}"
export TARGET_ROUTER_API_URL="${TARGET_ROUTER_API_URL:-http://localhost:8080}"
export TARGET_ROUTER_METRICS_URL="${TARGET_ROUTER_METRICS_URL:-http://localhost:9190/metrics}"

# MongoDB and chat-ui
export MONGODB_DATA_PATH="${MONGODB_DATA_PATH:-/var/lib/mongodb}"
export MONGODB_URL="${MONGODB_URL:-mongodb://127.0.0.1:27017}"
export MONGODB_DB_NAME="${MONGODB_DB_NAME:-chat-ui}"
export CHATUI_PORT="${CHATUI_PORT:-5173}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8801/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-placeholder}"
export PUBLIC_APP_NAME="${PUBLIC_APP_NAME:-HuggingChat}"
export PUBLIC_APP_ASSETS="${PUBLIC_APP_ASSETS:-chatui}"
export COOKIE_SECURE="${COOKIE_SECURE:-false}"
export COOKIE_NAME="${COOKIE_NAME:-hf-chat}"
export COOKIE_SAMESITE="${COOKIE_SAMESITE:-lax}"
export APP_BASE_URL="${APP_BASE_URL:-http://localhost:${CHATUI_PORT}}"

# Open WebUI and Pipelines
export OPENWEBUI_PORT="${OPENWEBUI_PORT:-3001}"
export OPENWEBUI_DATA_DIR="${OPENWEBUI_DATA_DIR:-/var/lib/openwebui}"
export PIPELINES_PORT="${PIPELINES_PORT:-9099}"
export TARGET_OPENWEBUI_URL="${TARGET_OPENWEBUI_URL:-http://localhost:${OPENWEBUI_PORT}}"

# LLM-Katan (lightweight LLM server)
export LLMKATAN_PORT="${LLMKATAN_PORT:-8002}"
export LLMKATAN_MODEL="${LLMKATAN_MODEL:-/app/models/Qwen/Qwen3-0.6B}"
export LLMKATAN_SERVED_MODEL_NAME="${LLMKATAN_SERVED_MODEL_NAME:-qwen3}"

# Display startup banner
cat << 'EOF'
============================================
  vLLM Semantic Router - Stack
============================================

   _____ ________  ___   ___   _   _  _____ _____ _____
  /  ___|  ___|  \/  | / _ \ | \ | ||_   _|_   _/  __ \
  \ `--.| |__ | .  . |/ /_\ \|  \| |  | |   | | | /  \/
   `--. \  __|| |\/| ||  _  || . ` |  | |   | | | |
  /\__/ / |___| |  | || | | || |\  |  | |  _| |_| \__/\
  \____/\____/\_|  |_/\_| |_/\_| \_/  \_/  \___/ \____/

            ROUTER  -  Stack Edition
============================================
EOF

echo ""
echo "Endpoints:"
echo "  API Gateway:       http://0.0.0.0:${ENVOY_LISTEN_PORT}"
echo "  Dashboard:         http://0.0.0.0:${DASHBOARD_PORT}"
echo "  HuggingChat:       http://0.0.0.0:${CHATUI_PORT}"
echo "  Open WebUI:        http://0.0.0.0:${OPENWEBUI_PORT}"
echo "  Pipelines:         http://0.0.0.0:${PIPELINES_PORT}"
echo "  LLM-Katan:         http://0.0.0.0:${LLMKATAN_PORT} (model: ${LLMKATAN_SERVED_MODEL_NAME})"
echo "  Prometheus:        http://0.0.0.0:9090"
echo "  Grafana:           http://0.0.0.0:3000 (${GF_SECURITY_ADMIN_USER}/${GF_SECURITY_ADMIN_PASSWORD})"
echo "  Jaeger:            http://0.0.0.0:16686"
echo "  Envoy Admin:       http://0.0.0.0:${ENVOY_ADMIN_PORT}"
echo ""
echo "Configuration: ${CONFIG_FILE}"
echo "============================================"

# Check config file
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "WARNING: Config file not found at ${CONFIG_FILE}"
fi

# Check models directory
if [ -d "/app/models" ]; then
    MODEL_COUNT=$(find /app/models -type f \( -name "*.bin" -o -name "*.safetensors" -o -name "*.json" \) 2>/dev/null | wc -l)
    echo "Models: Found ${MODEL_COUNT} model file(s) in /app/models"
else
    echo "WARNING: Mount models with: -v /path/to/models:/app/models"
fi

# Generate Envoy configuration from template
echo ""
echo "Generating configurations..."
if [ -f /etc/envoy/envoy.template.yaml ]; then
    # shellcheck disable=SC2016 # envsubst requires literal variable names in single quotes
    envsubst '${ENVOY_LISTEN_PORT} ${ENVOY_ADMIN_PORT} ${EXTPROC_HOST} ${EXTPROC_PORT}' \
        < /etc/envoy/envoy.template.yaml \
        > /etc/envoy/envoy.yaml
    echo "  -> Envoy config: /etc/envoy/envoy.yaml"
fi

# Allow custom config override
if [ -f /app/config/envoy.yaml ]; then
    cp /app/config/envoy.yaml /etc/envoy/envoy.yaml
    echo "  -> Using custom Envoy config from /app/config/envoy.yaml"
fi

# Create required directories
mkdir -p /var/log/supervisor /var/lib/grafana /var/lib/prometheus "${MONGODB_DATA_PATH}" "${OPENWEBUI_DATA_DIR}"
chown -R nobody:nogroup /var/lib/grafana 2>/dev/null || true

echo ""
echo "Starting all services..."
echo "============================================"

# Execute supervisord
exec "$@"
