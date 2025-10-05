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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

echo -e "${BLUE}==================================================================${NC}"
echo -e "${BLUE}  Stopping Observability Stack${NC}"
echo -e "${BLUE}==================================================================${NC}"
echo ""

# Stop services
log_info "Stopping observability services..."

# Try stopping local mode containers first
if docker ps -a --format '{{.Names}}' | grep -qE '^(prometheus-local|grafana-local)$'; then
    log_info "Stopping local mode containers..."
    docker compose -f "${PROJECT_ROOT}/docker-compose.obs.yml" down
fi

# Also stop compose mode if running as part of main stack
if docker ps -a --format '{{.Names}}' | grep -qE '^(prometheus|grafana)$' && ! docker ps -a --format '{{.Names}}' | grep -q 'semantic-router'; then
    log_warn "Observability containers from main stack are running"
    log_info "Use 'docker compose down' to stop the full stack"
fi

echo ""
log_info "Observability stopped"
echo ""
