#!/bin/bash
#
# Live Classifier Log Viewer for Semantic Router
#
# This script shows the CLASSIFICATION API logs (not Envoy traffic)
# Perfect for demonstrating curl-examples.sh in real-time
#
# Usage: ./live-demo-classifier-logs.sh
#

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║       SEMANTIC ROUTER - CLASSIFICATION API LOG VIEWER                     ║${NC}"
echo -e "${BOLD}${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}📡 Watching semantic-router (classification API) logs...${NC}"
echo -e "${CYAN}Press Ctrl+C to stop${NC}"
echo ""
echo -e "${BOLD}Legend:${NC}"
echo -e "  ${GREEN}📊 METRICS${NC}         - API metrics and stats"
echo -e "  ${BLUE}🔍 CLASSIFIER${NC}      - Classification events"
echo -e "  ${MAGENTA}⚙️  SYSTEM${NC}         - System/initialization events"
echo -e "  ${YELLOW}📝 REQUEST${NC}        - API requests"
echo ""
echo -e "${BOLD}${CYAN}────────────────────────────────────────────────────────────────────────────${NC}"
echo ""
echo -e "${YELLOW}💡 Tip: Run this alongside curl-examples.sh to see classification in action!${NC}"
echo ""
echo -e "${BOLD}${CYAN}────────────────────────────────────────────────────────────────────────────${NC}"
echo ""

# Counter for events
event_count=0

# Tail logs and process all events
oc logs -n vllm-semantic-router-system deployment/semantic-router --container=semantic-router --follow --tail=5 2>/dev/null | while read -r line; do
    # Skip empty lines
    if [ -z "$line" ]; then
        continue
    fi

    # Get timestamp if available
    timestamp=$(echo "$line" | grep -o '"ts":"[^"]*"' | cut -d'"' -f4 | cut -d'T' -f2 | cut -d'.' -f1)

    # Get log level
    level=$(echo "$line" | grep -o '"level":"[^"]*"' | cut -d'"' -f4)

    # Get message
    msg=$(echo "$line" | grep -o '"msg":"[^"]*"' | cut -d'"' -f4)

    # Increment counter
    ((event_count++))

    # Skip health checks
    if echo "$line" | grep -qi "health check"; then
        continue
    fi

    # Highlight INITIALIZATION events
    if echo "$msg" | grep -qi "initializ\|starting\|listening\|loaded"; then
        echo -e "${MAGENTA}⚙️  [${timestamp}] SYSTEM:${NC} ${msg}"
        continue
    fi

    # Highlight CLASSIFICATION events
    if echo "$msg" | grep -qi "classif\|category\|intent"; then
        echo -e "${BLUE}🔍 [${timestamp}] CLASSIFIER:${NC} ${BOLD}${msg}${NC}"
        continue
    fi

    # Highlight METRICS/STATS
    if echo "$line" | grep -q '"event"'; then
        event=$(echo "$line" | grep -o '"event":"[^"]*"' | cut -d'"' -f4)
        echo -e "${GREEN}📊 [${timestamp}] METRICS:${NC} event=${event}"
        continue
    fi

    # Highlight REQUEST processing
    if echo "$msg" | grep -qi "request\|processing"; then
        echo -e "${YELLOW}📝 [${timestamp}] REQUEST:${NC} ${msg}"
        continue
    fi

    # Highlight ERRORS
    if [ "$level" = "error" ] || echo "$msg" | grep -qi "error\|fail"; then
        echo -e "${RED}❌ [${timestamp}] ERROR:${NC} ${BOLD}${RED}${msg}${NC}"
        continue
    fi

    # Show other interesting messages
    if echo "$msg" | grep -qi "model\|router\|server"; then
        echo -e "${CYAN}ℹ️  [${timestamp}] INFO:${NC} ${msg}"
    fi

    # Show raw JSON every 50 events for debugging (optional)
    # if [ $((event_count % 50)) -eq 0 ]; then
    #     echo -e "${CYAN}[DEBUG] Raw: ${line:0:100}...${NC}"
    # fi
done
