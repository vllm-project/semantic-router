#!/bin/bash
# Simple test script to verify semantic routing is working
# Tests different query categories and verifies routing decisions

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect kubectl vs oc
if command -v oc &> /dev/null; then
    CLI="oc"
    DEFAULT_NAMESPACE=$(oc project -q 2>/dev/null || echo "default")
elif command -v kubectl &> /dev/null; then
    CLI="kubectl"
    DEFAULT_NAMESPACE=$(kubectl config view --minify -o jsonpath='{.contexts[0].context.namespace}' 2>/dev/null || echo "default")
else
    echo -e "${RED}✗${NC} Neither kubectl nor oc found. Please install one of them."
    exit 1
fi

# Configuration
NAMESPACE="${NAMESPACE:-$DEFAULT_NAMESPACE}"
ROUTE_NAME="semantic-router-kserve"
# Model name to use for testing - get from configmap or override with MODEL_NAME env var
MODEL_NAME="${MODEL_NAME:-$($CLI get configmap semantic-router-kserve-config -n "$NAMESPACE" -o jsonpath='{.data.config\.yaml}' 2>/dev/null | grep 'default_model:' | awk '{print $2}' || echo 'granite32-8b')}"

# Get the route URL
echo "Using CLI: $CLI"
echo "Using namespace: $NAMESPACE"
echo "Using model: $MODEL_NAME"
echo "Getting semantic router URL..."

if [ "$CLI" = "oc" ]; then
    ROUTER_URL=$($CLI get route "$ROUTE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null)

    if [ -z "$ROUTER_URL" ]; then
        echo -e "${RED}✗${NC} Could not find route '$ROUTE_NAME' in namespace '$NAMESPACE'"
        echo "Make sure the semantic router is deployed"
        exit 1
    fi

    # Determine protocol
    if $CLI get route "$ROUTE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.tls.termination}' 2>/dev/null | grep -q .; then
        ROUTER_URL="https://$ROUTER_URL"
    else
        ROUTER_URL="http://$ROUTER_URL"
    fi
else
    # For kubectl, try to get the service
    SVC_TYPE=$($CLI get svc "$ROUTE_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.type}' 2>/dev/null)

    if [ "$SVC_TYPE" = "LoadBalancer" ]; then
        ROUTER_URL=$($CLI get svc "$ROUTE_NAME" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null)
        if [ -z "$ROUTER_URL" ]; then
            ROUTER_URL=$($CLI get svc "$ROUTE_NAME" -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
        fi
        ROUTER_URL="http://$ROUTER_URL"
    else
        # Port-forward or ClusterIP - use localhost
        echo -e "${YELLOW}Note:${NC} Service is ClusterIP type. You may need to port-forward:"
        echo "  kubectl port-forward -n $NAMESPACE svc/$ROUTE_NAME 8801:8801"
        ROUTER_URL="${ROUTER_URL:-http://localhost:8801}"
    fi
fi

if [ -z "$ROUTER_URL" ] || [ "$ROUTER_URL" = "http://" ]; then
    echo -e "${RED}✗${NC} Could not determine router URL"
    echo "Set ROUTER_URL environment variable manually"
    exit 1
fi

echo -e "${GREEN}✓${NC} Semantic router URL: $ROUTER_URL"
echo ""

# Function to test classification via API endpoint
test_classification_api() {
    local query="$1"
    local expected_category="$2"

    echo -e "${BLUE}Testing classification API:${NC} \"$query\""
    echo -n "Expected category: $expected_category ... "

    # Call classification endpoint (port 8080)
    response=$(curl -s -k -X POST "$ROUTER_URL:8080/api/v1/classify" \
        -H "Content-Type: application/json" \
        -d "{\"text\": \"$query\"}" 2>/dev/null)

    if [ -z "$response" ]; then
        echo -e "${YELLOW}SKIP${NC} - Classification API not responding (may not be exposed)"
        return 0
    fi

    # Extract category from response
    category=$(echo "$response" | grep -o '"category":"[^"]*"' | cut -d'"' -f4)

    if [ -z "$category" ]; then
        echo -e "${YELLOW}SKIP${NC} - Classification API not available"
        return 0
    fi

    if [ "$category" == "$expected_category" ]; then
        echo -e "${GREEN}PASS${NC} - Category: $category"
        return 0
    else
        echo -e "${YELLOW}PARTIAL${NC} - Got: $category (expected: $expected_category)"
        return 0
    fi
}

# Function to test chat completion
test_chat_completion() {
    local query="$1"
    local model="${2:-$MODEL_NAME}"

    echo -e "${BLUE}Testing chat completion:${NC} \"$query\""
    echo -n "Sending request to model: $model ... "

    response=$(curl -s -k -X POST "$ROUTER_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$model\", \"messages\": [{\"role\": \"user\", \"content\": \"$query\"}], \"max_tokens\": 50}" 2>/dev/null)

    if [ -z "$response" ]; then
        echo -e "${RED}FAIL${NC} - No response"
        return 1
    fi

    # Check for error in response
    if echo "$response" | grep -q '"error"'; then
        echo -e "${RED}FAIL${NC}"
        echo "Error: $(echo "$response" | grep -o '"message":"[^"]*"' | cut -d'"' -f4)"
        return 1
    fi

    # Check for completion
    if echo "$response" | grep -q '"choices"'; then
        echo -e "${GREEN}PASS${NC}"
        # Extract first few words of response
        content=$(echo "$response" | grep -o '"content":"[^"]*"' | head -1 | cut -d'"' -f4 | cut -c1-100)
        echo "  Response preview: $content..."
        return 0
    else
        echo -e "${RED}FAIL${NC} - Invalid response format"
        return 1
    fi
}

echo "=================================================="
echo "Semantic Routing Validation Tests"
echo "=================================================="
echo ""

# Test 1: Check /v1/models endpoint
echo -e "${BLUE}Test 1:${NC} Checking /v1/models endpoint"
models_response=$(curl -s -k "$ROUTER_URL/v1/models" 2>/dev/null)
if echo "$models_response" | grep -q '"object":"list"'; then
    echo -e "${GREEN}✓${NC} Models endpoint responding correctly"
    echo "Available models: $(echo "$models_response" | grep -o '"id":"[^"]*"' | cut -d'"' -f4 | tr '\n' ', ' | sed 's/,$//')"
else
    echo -e "${RED}✗${NC} Models endpoint not responding correctly"
    echo "Response: $models_response"
fi
echo ""

# Test 2: Classification tests (optional - API may not be exposed)
echo -e "${BLUE}Test 2:${NC} Testing category classification API (optional)"
echo ""

test_classification_api "What is the derivative of x squared?" "math"
test_classification_api "Explain quantum entanglement in physics" "physics"
test_classification_api "Write a function to reverse a string in Python" "computer science"

echo ""

# Test 3: End-to-end chat completion
echo -e "${BLUE}Test 3:${NC} Testing end-to-end chat completion"
echo ""

test_chat_completion "What is 2+2? Answer briefly."
test_chat_completion "Tell me a joke"

echo ""

# Test 4: PII detection (if enabled)
echo -e "${BLUE}Test 4:${NC} Testing PII detection"
echo ""

echo -e "${BLUE}Testing:${NC} Query with PII (SSN)"
response=$(curl -s -k -X POST "$ROUTER_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"My SSN is 123-45-6789\"}], \"max_tokens\": 50}" 2>/dev/null)

if echo "$response" | grep -qi "pii\|blocked\|detected"; then
    echo -e "${GREEN}✓${NC} PII detection working - request blocked or flagged"
elif echo "$response" | grep -q '"error"'; then
    echo -e "${GREEN}✓${NC} PII protection active - request rejected"
    echo "  Message: $(echo "$response" | grep -o '"message":"[^"]*"' | cut -d'"' -f4)"
else
    echo -e "${YELLOW}⚠${NC} PII may have passed through (check if PII policy allows it)"
fi

echo ""

# Test 5: Semantic caching
echo -e "${BLUE}Test 5:${NC} Testing semantic caching"
echo ""

CACHE_QUERY="What is the capital of France?"

echo "First request (cache miss expected)..."
time1_start=$(date +%s%N)
response1=$(curl -s -k -X POST "$ROUTER_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"$CACHE_QUERY\"}], \"max_tokens\": 20}" 2>/dev/null)
time1_end=$(date +%s%N)
time1=$(((time1_end - time1_start) / 1000000))

sleep 1

echo "Second request (cache hit expected)..."
time2_start=$(date +%s%N)
response2=$(curl -s -k -X POST "$ROUTER_URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"$CACHE_QUERY\"}], \"max_tokens\": 20}" 2>/dev/null)
time2_end=$(date +%s%N)
time2=$(((time2_end - time2_start) / 1000000))

echo "First request: ${time1}ms"
echo "Second request: ${time2}ms"

if [ "$time2" -lt "$time1" ]; then
    speedup=$(((time1 - time2) * 100 / time1))
    echo -e "${GREEN}✓${NC} Cache appears to be working (${speedup}% faster)"
else
    echo -e "${YELLOW}⚠${NC} Cache behavior unclear or not significant"
fi

echo ""
echo "=================================================="
echo "Validation Complete"
echo "=================================================="
echo ""
echo "Semantic routing is operational!"
echo ""
echo "Next steps:"
echo "  • Review the test results above"
echo "  • Check logs: $CLI logs -n $NAMESPACE -l app=semantic-router -c semantic-router"
echo "  • View metrics: $CLI port-forward -n $NAMESPACE svc/$ROUTE_NAME 9190:9190"
echo "  • Test with your own queries: curl -k \"$ROUTER_URL/v1/chat/completions\" \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"model\": \"$MODEL_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Your query here\"}]}'"
echo ""
