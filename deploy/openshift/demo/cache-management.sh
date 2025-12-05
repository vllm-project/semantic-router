#!/bin/bash
#
# Cache Management Helper for Semantic Router Demo
#
# This script helps manage the in-memory cache for demo purposes
#

NAMESPACE="vllm-semantic-router-system"

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

function print_header() {
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

function check_cache_status() {
    print_header "CACHE STATUS"

    echo -e "${CYAN}Checking recent cache activity...${NC}"
    echo ""

    # Get recent cache hits
    cache_hits=$(oc logs -n $NAMESPACE deployment/semantic-router --tail=50 2>/dev/null | grep -c "cache_hit")

    if [ "$cache_hits" -gt 0 ]; then
        echo -e "${GREEN}✅ Cache is active${NC}"
        echo -e "   Recent cache hits: ${BOLD}${cache_hits}${NC}"
        echo ""
        echo -e "${CYAN}Recent cached queries:${NC}"
        oc logs -n $NAMESPACE deployment/semantic-router --tail=100 2>/dev/null | \
            grep "cache_hit" | grep -o '"query":"[^"]*"' | cut -d'"' -f4 | sort -u | tail -5 | \
            while read -r query; do
                echo -e "   💾 ${query}"
            done
    else
        echo -e "${YELLOW}⚠️  No recent cache activity${NC}"
        echo -e "   Cache may be empty or recently cleared"
    fi
    echo ""
}

function clear_cache() {
    print_header "CLEAR CACHE"

    echo -e "${YELLOW}This will restart the semantic-router deployment to clear the in-memory cache.${NC}"
    echo -e "${YELLOW}The pod will be recreated automatically (takes ~30 seconds).${NC}"
    echo ""
    read -r -p "Are you sure you want to clear the cache? (y/N): " confirm

    if [[ $confirm =~ ^[Yy]$ ]]; then
        echo ""
        echo -e "${CYAN}Clearing cache by restarting deployment...${NC}"
        oc rollout restart deployment/semantic-router -n $NAMESPACE

        echo ""
        echo -e "${CYAN}Waiting for new pod to be ready...${NC}"
        oc rollout status deployment/semantic-router -n $NAMESPACE --timeout=60s

        echo ""
        echo -e "${GREEN}✅ Cache cleared! New pod is running.${NC}"
        echo ""

        # Show new pod
        echo -e "${CYAN}New pod:${NC}"
        oc get pods -n $NAMESPACE -l app=semantic-router
        echo ""
    else
        echo -e "${YELLOW}Cache clear cancelled.${NC}"
    fi
}

function demo_cache_workflow() {
    print_header "DEMO CACHE WORKFLOW"

    echo -e "${BOLD}Demo Scenario: Show Cache in Action${NC}"
    echo ""
    echo -e "${CYAN}Step 1: Clear the cache${NC}"
    echo -e "   ./deploy/openshift/demo/cache-management.sh clear"
    echo ""
    echo -e "${CYAN}Step 2: Start the log viewer (Terminal 1)${NC}"
    echo -e "   ./deploy/openshift/demo/live-demo-logs.sh"
    echo ""
    echo -e "${CYAN}Step 3: Run the classification test ONCE (Terminal 2)${NC}"
    echo -e "   python3 deploy/openshift/demo/demo-classification-test.py"
    echo -e "   ${YELLOW}→ You'll see NO cache hits (first time)${NC}"
    echo ""
    echo -e "${CYAN}Step 4: Run the classification test AGAIN (Terminal 2)${NC}"
    echo -e "   python3 deploy/openshift/demo/demo-classification-test.py"
    echo -e "   ${GREEN}→ You'll see CACHE HITS for all 10 prompts! 💾${NC}"
    echo ""
    echo -e "${CYAN}Step 5: Point out in the logs:${NC}"
    echo -e "   - First run: Queries go to the model (slower, ~3-4 seconds)"
    echo -e "   - Second run: Queries hit cache (faster, ~400ms)"
    echo -e "   - Similarity scores: ${BOLD}~0.99999${NC} (semantic matching)"
    echo ""
    echo -e "${BOLD}${GREEN}Key Demo Point:${NC}"
    echo -e "   Cache uses ${BOLD}semantic similarity${NC}, not exact string matching!"
    echo -e "   Similar questions will also hit the cache."
    echo ""
}

# Main menu
case "${1:-menu}" in
    status)
        check_cache_status
        ;;
    clear)
        clear_cache
        ;;
    demo)
        demo_cache_workflow
        ;;
    menu|*)
        print_header "SEMANTIC ROUTER - CACHE MANAGEMENT"
        echo ""
        echo -e "${BOLD}Commands:${NC}"
        echo ""
        echo -e "  ${CYAN}./deploy/openshift/demo/cache-management.sh status${NC}"
        echo -e "     Check current cache status and recent activity"
        echo ""
        echo -e "  ${CYAN}./deploy/openshift/demo/cache-management.sh clear${NC}"
        echo -e "     Clear the cache by restarting semantic-router deployment"
        echo ""
        echo -e "  ${CYAN}./deploy/openshift/demo/cache-management.sh demo${NC}"
        echo -e "     Show demo workflow for cache feature"
        echo ""
        ;;
esac
