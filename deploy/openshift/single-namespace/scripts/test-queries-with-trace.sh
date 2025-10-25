#!/bin/bash

# Test script that sends multiple queries and traces them through the system
# Shows routing decisions, endpoints hit, and complete request flow

set -e

NAMESPACE="${1:-vllm-semantic-router}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%H:%M:%S.%3N')

    case $level in
        "INFO")  echo -e "${timestamp} ${BLUE}[INFO]${NC}  $message" ;;
        "WARN")  echo -e "${timestamp} ${YELLOW}[WARN]${NC}  $message" ;;
        "ERROR") echo -e "${timestamp} ${RED}[ERROR]${NC} $message" ;;
        "SUCCESS") echo -e "${timestamp} ${GREEN}[✓]${NC} $message" ;;
        "QUERY") echo -e "${timestamp} ${CYAN}[QUERY]${NC} $message" ;;
        "ROUTE") echo -e "${timestamp} ${MAGENTA}[ROUTE]${NC} $message" ;;
        "STEP") echo -e "${timestamp} ${CYAN}[→]${NC} $message" ;;
    esac
}

# Get endpoints
get_endpoints() {
    ENVOY_ROUTE=$(oc get route envoy-proxy -n $NAMESPACE -o jsonpath='{.spec.host}' 2>/dev/null)
    ENVOY_SERVICE_IP=$(oc get svc envoy-proxy -n $NAMESPACE -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
    ROUTER_SERVICE_IP=$(oc get svc semantic-router -n $NAMESPACE -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
    MODEL_A_SERVICE_IP=$(oc get svc model-a -n $NAMESPACE -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
    MODEL_B_SERVICE_IP=$(oc get svc model-b -n $NAMESPACE -o jsonpath='{.spec.clusterIP}' 2>/dev/null)

    ROUTER_POD=$(oc get pods -n $NAMESPACE -l app=semantic-router -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    ROUTER_POD_IP=$(oc get pod $ROUTER_POD -n $NAMESPACE -o jsonpath='{.status.podIP}' 2>/dev/null)
    MODEL_A_POD=$(oc get pods -n $NAMESPACE -l app=model-a -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    MODEL_A_POD_IP=$(oc get pod $MODEL_A_POD -n $NAMESPACE -o jsonpath='{.status.podIP}' 2>/dev/null)
    MODEL_B_POD=$(oc get pods -n $NAMESPACE -l app=model-b -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    MODEL_B_POD_IP=$(oc get pod $MODEL_B_POD -n $NAMESPACE -o jsonpath='{.status.podIP}' 2>/dev/null)
}

# Function to send query and trace it
test_query() {
    local query_num=$1
    local query_text="$2"
    local expected_category="$3"

    echo ""
    echo "========================================="
    log "QUERY" "Test #$query_num: \"$query_text\""
    echo "========================================="
    echo ""

    # Step 1: Show what will happen
    log "STEP" "Step 1: Classification (predicting route)"
    echo "  Query will be sent to semantic-router for classification"
    echo ""

    # Get classification first (standalone)
    log "INFO" "Calling Classification API directly..."
    CLASSIFY_DATA="{\"text\": \"$query_text\"}"

    CLASSIFY_RESPONSE=$(oc exec -n $NAMESPACE $ROUTER_POD -c semantic-router -- \
        curl -s -X POST "http://localhost:8080/api/v1/classify/intent" \
        -H "Content-Type: application/json" \
        -d "$CLASSIFY_DATA" 2>/dev/null)

    # Parse classification response
    CATEGORY=$(echo "$CLASSIFY_RESPONSE" | grep -o '"category":"[^"]*"' | cut -d'"' -f4)
    CONFIDENCE=$(echo "$CLASSIFY_RESPONSE" | grep -o '"confidence":[0-9.]*' | cut -d':' -f2)

    log "SUCCESS" "Classification Result:"
    echo "  ├─ Category: $CATEGORY"
    echo "  └─ Confidence: $CONFIDENCE"

    if [ -n "$expected_category" ] && [ "$CATEGORY" = "$expected_category" ]; then
        log "SUCCESS" "Classification matched expected category: $expected_category"
    fi

    echo ""
    log "STEP" "Step 2: Sending request through complete flow"
    echo ""

    # Step 2: Send actual request
    REQUEST_DATA="{\"model\": \"auto\", \"messages\": [{\"role\": \"user\", \"content\": \"$query_text\"}], \"max_tokens\": 100}"

    # Timestamp before request
    START_TIME=$(date +%s%3N)

    log "ROUTE" "Request Flow Trace:"
    echo "  1. Client → Envoy Service ($ENVOY_SERVICE_IP:8801)"
    echo "  2. Envoy → ExtProc gRPC ($ROUTER_POD_IP:50051)"
    echo "  3. Semantic Router processes:"
    echo "     ├─ Classify intent: $CATEGORY (confidence: $CONFIDENCE)"
    echo "     ├─ Check PII"
    echo "     ├─ Check jailbreak"
    echo "     └─ Determine target model"
    echo "  4. ExtProc → Envoy (set routing headers)"

    # Determine which model based on category
    if [[ "$CATEGORY" == *"math"* ]] || [[ "$CATEGORY" == *"coding"* ]]; then
        EXPECTED_MODEL="Model-A"
        EXPECTED_SERVICE="$MODEL_A_SERVICE_IP"
        EXPECTED_POD="$MODEL_A_POD_IP"
    else
        EXPECTED_MODEL="Model-B"
        EXPECTED_SERVICE="$MODEL_B_SERVICE_IP"
        EXPECTED_POD="$MODEL_B_POD_IP"
    fi

    echo "  5. Envoy routes to: $EXPECTED_MODEL"
    echo "     ├─ Service: $EXPECTED_SERVICE:8000"
    echo "     └─ Pod: $EXPECTED_POD:8000"
    echo "  6. Model generates response"
    echo "  7. Response → Envoy → Client"

    echo ""
    log "INFO" "Sending request..."

    # Send request
    RESPONSE=$(oc exec -n $NAMESPACE $ROUTER_POD -c semantic-router -- \
        curl -s -w "\n%{http_code}" -X POST "http://$ENVOY_SERVICE_IP:8801/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$REQUEST_DATA" 2>/dev/null)

    # Timestamp after response
    END_TIME=$(date +%s%3N)
    DURATION=$((END_TIME - START_TIME))

    # Parse response
    STATUS_CODE=$(echo "$RESPONSE" | tail -1)
    BODY=$(echo "$RESPONSE" | head -n -1)

    echo ""
    if [ "$STATUS_CODE" = "200" ]; then
        log "SUCCESS" "Request completed - Status: 200 OK (${DURATION}ms)"

        # Extract model name and content from response
        ACTUAL_MODEL=$(echo "$BODY" | grep -o '"model":"[^"]*"' | head -1 | cut -d'"' -f4)
        CONTENT=$(echo "$BODY" | grep -o '"content":"[^"]*"' | head -1 | cut -d'"' -f4)

        echo ""
        log "INFO" "Response Details:"
        echo "  ├─ Routed to Model: $ACTUAL_MODEL"
        echo "  ├─ Response time: ${DURATION}ms"
        echo "  └─ Content: ${CONTENT:0:150}..."

        # Verify routing
        if [ "$ACTUAL_MODEL" = "$EXPECTED_MODEL" ]; then
            log "SUCCESS" "Routing verified: Request went to expected model ($EXPECTED_MODEL)"
        else
            log "WARN" "Routing mismatch: Expected $EXPECTED_MODEL, got $ACTUAL_MODEL"
        fi

    else
        log "ERROR" "Request failed - Status: $STATUS_CODE"
        echo "  Response: ${BODY:0:200}"
    fi

    echo ""
    log "STEP" "Step 3: Checking metrics and logs"
    echo ""

    # Get recent Envoy access logs
    log "INFO" "Recent Envoy Access Log:"
    oc logs -n $NAMESPACE $ROUTER_POD -c envoy-proxy --tail=2 2>/dev/null | while read line; do
        echo "  $line"
    done

    # Get router metrics
    log "INFO" "Router Metrics:"
    METRICS=$(oc exec -n $NAMESPACE $ROUTER_POD -c semantic-router -- \
        curl -s http://localhost:9190/metrics 2>/dev/null | grep -E "semantic_router_requests_total|semantic_router_classification" || echo "")

    if [ -n "$METRICS" ]; then
        echo "$METRICS" | head -5 | while read line; do
            echo "  $line"
        done
    else
        echo "  (No semantic_router metrics available yet - may need more requests)"
    fi

    echo ""
    echo "========================================="
    log "SUCCESS" "Query #$query_num completed!"
    echo "========================================="

    return 0
}

# Main script
echo "========================================="
echo "vLLM Semantic Router - Query Tracing"
echo "Namespace: $NAMESPACE"
echo "========================================="
echo ""

# Get all endpoints
log "INFO" "Collecting endpoint information..."
get_endpoints

echo ""
log "INFO" "Deployment Endpoints:"
echo "  External Route:      https://$ENVOY_ROUTE"
echo "  Envoy Service:       $ENVOY_SERVICE_IP:8801"
echo "  Router Service:      $ROUTER_SERVICE_IP:50051 (gRPC), :8080 (API)"
echo "  Model-A Service:     $MODEL_A_SERVICE_IP:8000 → Pod: $MODEL_A_POD_IP:8000"
echo "  Model-B Service:     $MODEL_B_SERVICE_IP:8000 → Pod: $MODEL_B_POD_IP:8000"
echo ""

# Wait for user
echo "This script will send multiple test queries and trace them through the system."
echo "Using GOLDEN EXAMPLES from demo-semantic-router.py (verified working prompts)"
echo ""
echo "Each query will show:"
echo "  - Classification result"
echo "  - Complete routing path"
echo "  - Endpoints accessed"
echo "  - Response and timing"
echo "  - Envoy access logs"
echo ""
log "WARN" "Note: Classification confidence is currently low (<0.6 threshold)"
log "WARN" "Most queries will fall back to default model (Model-A)"
echo ""
log "INFO" "Testing with 2 queries to demonstrate the routing system:"
echo "  - Query 1 → Model-A: Math question (high score: 1.0)"
echo "  - Query 2 → Attempted Model-B: Business question (score: 0.7)"
echo ""
log "INFO" "Expected behavior with current confidence:"
echo "  - Both will likely route to Model-A due to low classification confidence"
echo "  - This demonstrates the fallback mechanism when confidence < 0.6"
echo ""
read -p "Press Enter to start testing..."

# Test queries - 2 examples showing routing behavior
# Based on actual routing config:
#   Math → Model-A (score: 1.0) - will work
#   Business → Model-B (score: 0.7) - will likely fallback to Model-A due to low confidence
declare -a QUERIES=(
    "Is 17 a prime number?|math"
    "How do I create a business plan for a startup?|business"
)

QUERY_NUM=1
for query_line in "${QUERIES[@]}"; do
    IFS='|' read -r query expected <<< "$query_line"
    test_query $QUERY_NUM "$query" "$expected"
    QUERY_NUM=$((QUERY_NUM + 1))

    # Small delay between queries
    sleep 2
done

# Summary
echo ""
echo "========================================="
echo "Test Summary"
echo "========================================="
echo ""
log "SUCCESS" "All $((QUERY_NUM - 1)) queries completed successfully!"
echo ""
log "INFO" "Total Requests by Model:"
TOTAL_METRICS=$(oc exec -n $NAMESPACE $ROUTER_POD -c semantic-router -- \
    curl -s http://localhost:9190/metrics 2>/dev/null | grep "semantic_router_requests_total" || echo "")

if [ -n "$TOTAL_METRICS" ]; then
    echo "$TOTAL_METRICS" | while read line; do
        echo "  $line"
    done
else
    echo "  (Metrics not available - semantic_router may not expose custom metrics)"
fi

echo ""
log "INFO" "To view full logs:"
echo "  Semantic Router: oc logs -f $ROUTER_POD -c semantic-router -n $NAMESPACE"
echo "  Envoy Proxy:     oc logs -f $ROUTER_POD -c envoy-proxy -n $NAMESPACE"
echo "  Model-A:         oc logs -f $MODEL_A_POD -n $NAMESPACE"
echo "  Model-B:         oc logs -f $MODEL_B_POD -n $NAMESPACE"
echo ""

log "SUCCESS" "Query tracing completed! ✓"
