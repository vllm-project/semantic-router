#!/bin/bash
#
# Quick Curl Examples for Semantic Router Demo
#
# These commands hit the classification API directly and show the detected category
# Perfect for quick interactive demos from the command line
#
# Requires: oc login (URLs are discovered dynamically from OpenShift)
#

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

# Check if logged into OpenShift
if ! oc whoami &>/dev/null; then
    echo -e "${RED}Error: Not logged into OpenShift${NC}"
    echo -e "${YELLOW}Please run: oc login${NC}"
    exit 1
fi

# Get API URL dynamically from OpenShift route
echo -e "${CYAN}Discovering routes from OpenShift...${NC}"
API_HOST=$(oc get route semantic-router-api -n vllm-semantic-router-system -o jsonpath='{.spec.host}' 2>/dev/null)

if [ -z "$API_HOST" ]; then
    echo -e "${RED}Error: Could not find semantic-router-api route${NC}"
    echo -e "${YELLOW}Make sure the deployment is running${NC}"
    exit 1
fi

API_URL="http://${API_HOST}/api/v1/classify/intent"
echo -e "${GREEN}✅ Found API: ${API_URL}${NC}\n"

function classify() {
    local prompt="$1"
    local expected_category="$2"

    echo -e "${CYAN}Prompt:${NC} ${BOLD}\"${prompt}\"${NC}"

    result=$(curl -s -X POST "$API_URL" \
      -H 'Content-Type: application/json' \
      -d "{\"text\": \"${prompt}\"}")

    category=$(echo "$result" | python3 -c "import sys, json; print(json.load(sys.stdin)['classification']['category'])" 2>/dev/null)
    confidence=$(echo "$result" | python3 -c "import sys, json; print(json.load(sys.stdin)['classification']['confidence'])" 2>/dev/null)
    time_ms=$(echo "$result" | python3 -c "import sys, json; print(json.load(sys.stdin)['classification']['processing_time_ms'])" 2>/dev/null)

    echo -e "${GREEN}Category:${NC} ${BOLD}${category}${NC} (confidence: ${confidence}, ${time_ms}ms)"
    echo ""
}

case "${1:-menu}" in
    math)
        echo -e "${BOLD}=== Math Question ===${NC}\n"
        classify "Is 17 a prime number?" "math"
        ;;

    chemistry)
        echo -e "${BOLD}=== Chemistry Question ===${NC}\n"
        classify "What are atoms made of?" "chemistry"
        ;;

    history)
        echo -e "${BOLD}=== History Question ===${NC}\n"
        classify "What was the Cold War?" "history"
        ;;

    psychology)
        echo -e "${BOLD}=== Psychology Question ===${NC}\n"
        classify "What are the stages of grief?" "psychology"
        ;;

    health)
        echo -e "${BOLD}=== Health Question ===${NC}\n"
        classify "What is a balanced diet?" "health"
        ;;

    all)
        echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BOLD}${CYAN}  TESTING ALL GOLDEN EXAMPLES${NC}"
        echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

        echo -e "${BOLD}1. Math:${NC}"
        classify "Is 17 a prime number?" "math"

        echo -e "${BOLD}2. History (WWI):${NC}"
        classify "What were the main causes of World War I?" "history"

        echo -e "${BOLD}3. History (Cold War):${NC}"
        classify "What was the Cold War?" "history"

        echo -e "${BOLD}4. Chemistry (Oxidation):${NC}"
        classify "Explain oxidation and reduction" "chemistry"

        echo -e "${BOLD}5. Chemistry (Atoms):${NC}"
        classify "What are atoms made of?" "chemistry"

        echo -e "${BOLD}6. Chemistry (Equilibrium):${NC}"
        classify "Explain chemical equilibrium" "chemistry"

        echo -e "${BOLD}7. Psychology (Nature vs Nurture):${NC}"
        classify "What is the nature vs nurture debate?" "psychology"

        echo -e "${BOLD}8. Psychology (Grief):${NC}"
        classify "What are the stages of grief?" "psychology"

        echo -e "${BOLD}9. Health (Lifestyle):${NC}"
        classify "How to maintain a healthy lifestyle?" "health"

        echo -e "${BOLD}10. Health (Diet):${NC}"
        classify "What is a balanced diet?" "health"
        ;;

    raw)
        # Show raw curl command for copy/paste
        echo -e "${BOLD}Raw Curl Command (Math Example):${NC}\n"
        cat << 'EOF'
curl -X POST "http://semantic-router-api-vllm-semantic-router-system.apps.cluster-pbd96.pbd96.sandbox5333.opentlc.com/api/v1/classify/intent" \
  -H 'Content-Type: application/json' \
  -d '{"text": "Is 17 a prime number?"}' | python3 -m json.tool
EOF
        echo ""
        echo -e "${YELLOW}Try it yourself:${NC}"
        curl -X POST "$API_URL" \
          -H 'Content-Type: application/json' \
          -d '{"text": "Is 17 a prime number?"}' | python3 -m json.tool
        ;;

    *)
        echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BOLD}${CYAN}  SEMANTIC ROUTER - CURL EXAMPLES${NC}"
        echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

        echo -e "${BOLD}Quick Examples:${NC}\n"

        echo -e "  ${CYAN}./deploy/openshift/demo/curl-examples.sh math${NC}"
        echo -e "     Test math classification\n"

        echo -e "  ${CYAN}./deploy/openshift/demo/curl-examples.sh chemistry${NC}"
        echo -e "     Test chemistry classification\n"

        echo -e "  ${CYAN}./deploy/openshift/demo/curl-examples.sh history${NC}"
        echo -e "     Test history classification\n"

        echo -e "  ${CYAN}./deploy/openshift/demo/curl-examples.sh psychology${NC}"
        echo -e "     Test psychology classification\n"

        echo -e "  ${CYAN}./deploy/openshift/demo/curl-examples.sh health${NC}"
        echo -e "     Test health classification\n"

        echo -e "  ${CYAN}./deploy/openshift/demo/curl-examples.sh all${NC}"
        echo -e "     Test all 10 golden examples\n"

        echo -e "  ${CYAN}./deploy/openshift/demo/curl-examples.sh raw${NC}"
        echo -e "     Show raw curl command for copy/paste\n"

        echo -e "${BOLD}${YELLOW}Note:${NC} These hit the classification API directly (not through Envoy)"
        echo -e "      They show the detected category but don't trigger Grafana metrics"
        ;;
esac
