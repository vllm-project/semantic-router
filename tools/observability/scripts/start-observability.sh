#!/usr/bin/env bash
# start-observability.sh
#
# Starts Prometheus and Grafana using Docker Compose
# 
# This script starts observability stack to monitor semantic-router.
# It supports two modes:
#   - Local mode: Router running natively, observability in Docker (network_mode: host)
#   - Compose mode: All services in Docker (uses semantic-network)
#
# Prerequisites:
#   - Docker and Docker Compose installed and running
#
# Usage:
#   ./scripts/start-observability.sh [local|compose]
#
# To stop:
#   ./scripts/stop-observability.sh

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
# SCRIPT_DIR points to tools/observability/scripts
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Determine repository root robustly
if command -v git >/dev/null 2>&1 && git -C "$SCRIPT_DIR" rev-parse --show-toplevel >/dev/null 2>&1; then
    PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
else
    # Fallback: three levels up from scripts -> repo root
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
fi

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_debug() { echo -e "${BLUE}[DEBUG]${NC} $1"; }

# Parse mode argument
MODE="${1:-local}"

case "${MODE}" in
    local)
        log_info "Starting observability in LOCAL mode (router on host, observability in Docker)"
        COMPOSE_FILE="${PROJECT_ROOT}/tools/observability/docker-compose.obs.yml"
        ;;
    compose)
        log_info "Starting observability in COMPOSE mode (all services in Docker)"
        COMPOSE_FILE="${PROJECT_ROOT}/deploy/docker-compose/docker-compose.yml"
        ;;
    *)
        log_error "Invalid mode: ${MODE}"
        log_info "Usage: $0 [local|compose]"
        log_info "  local   - Router on host, observability in Docker (default)"
        log_info "  compose - All services in Docker (uses main docker-compose.yml)"
        exit 1
        ;;
esac

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed or not in PATH"
    log_info "Please install Docker Desktop: https://www.docker.com/products/docker-desktop"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    log_error "Docker daemon is not running"
    log_info "Please start Docker Desktop"
    exit 1
fi

log_info "Starting services..."
log_debug "Command: docker compose -f \"${COMPOSE_FILE}\" up -d"

docker compose -f "${COMPOSE_FILE}" up -d

# Wait for services to become healthy
log_info "Waiting for services to become healthy..."
sleep 10

# Check service status
if [[ "${MODE}" == "local" ]]; then
    PROM_CONTAINER="prometheus-local"
    GRAF_CONTAINER="grafana-local"
else
    PROM_CONTAINER="prometheus"
    GRAF_CONTAINER="grafana"
fi

if docker ps --format '{{.Names}}' | grep -q "^${PROM_CONTAINER}$"; then
    log_info "✓ Prometheus is running at http://localhost:9090"
else
    log_warn "⚠ Prometheus not running"
    log_info "  Check logs: docker logs ${PROM_CONTAINER}"
fi

if docker ps --format '{{.Names}}' | grep -q "^${GRAF_CONTAINER}$"; then
    log_info "✓ Grafana is running at http://localhost:3000"
    log_info "  Default credentials: admin / admin"
else
    log_warn "⚠ Grafana not running"
    log_info "  Check logs: docker logs ${GRAF_CONTAINER}"
fi

echo ""
log_info "==================================================================="
log_info "Observability stack started successfully in ${MODE^^} mode!"
log_info "==================================================================="
echo ""

if [[ "${MODE}" == "local" ]]; then
    log_info "Next steps:"
    log_info "  1. Start semantic-router on localhost:9190"
    log_info "  2. Open Prometheus: http://localhost:9090/targets"
    log_info "  3. Open Grafana: http://localhost:3000"
    log_info "  4. View dashboard: LLM Router Metrics"
else
    log_info "Next steps:"
    log_info "  1. Ensure semantic-router is running in Docker"
    log_info "  2. Open Prometheus: http://localhost:9090/targets"
    log_info "  3. Open Grafana: http://localhost:3000"
    log_info "  4. View dashboard: LLM Router Metrics"
fi

echo ""
log_info "Useful commands:"
if [[ "${MODE}" == "local" ]]; then
    log_info "  - Check status: docker compose -f docker-compose.obs.yml ps"
    log_info "  - View logs: docker compose -f docker-compose.obs.yml logs -f"
else
    log_info "  - Check status: docker compose ps"
    log_info "  - View logs: docker compose logs prometheus grafana -f"
fi
log_info "  - Stop services: make stop-observability"
echo ""
