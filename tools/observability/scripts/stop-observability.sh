#!/usr/bin/env bash
# stop-observability.sh
#
# Stops and removes observability Docker containers using Docker Compose.
#
# Usage:
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

echo -e "${BLUE}==================================================================${NC}"
echo -e "${BLUE}  Stopping Observability Stack${NC}"
echo -e "${BLUE}==================================================================${NC}"
echo ""

# Helpers
container_exists() {
    # Returns 0 if a container with the given name exists (in any state)
    docker ps -a --format '{{.Names}}' | grep -Fxq "$1"
}

any_container_exists() {
    # Returns 0 if any of the provided container names exist
    local name
    for name in "$@"; do
        if container_exists "$name"; then
            return 0
        fi
    done
    return 1
}

# Stop services
log_info "Stopping observability services..."

# Try stopping local mode containers first
LOCAL_MODE_RUNNING=false
if any_container_exists "prometheus-local" "grafana-local"; then
    LOCAL_MODE_RUNNING=true
fi
if [ "${LOCAL_MODE_RUNNING}" = true ]; then
    log_info "Stopping local mode containers..."
    docker compose -f "${PROJECT_ROOT}/docker-compose.obs.yml" down
fi

# Also stop compose mode if running as part of main stack
COMPOSE_O11Y_RUNNING=false
if any_container_exists "prometheus" "grafana"; then
    COMPOSE_O11Y_RUNNING=true
fi

ROUTER_RUNNING=false
if container_exists "semantic-router"; then
    ROUTER_RUNNING=true
fi

if [ "${COMPOSE_O11Y_RUNNING}" = true ] && [ "${ROUTER_RUNNING}" = false ]; then
    log_warn "Observability containers from main stack are running"
    log_info "Use 'docker compose down' to stop the full stack"
fi

echo ""
log_info "Observability stopped"
echo ""
