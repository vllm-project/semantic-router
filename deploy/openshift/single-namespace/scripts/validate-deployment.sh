#!/bin/bash

# Validation script for vLLM Semantic Router deployment
# Tests all endpoints and traces request flow through the system

set -e

NAMESPACE="${1:-vllm-semantic-router}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
log() {
    local level=$1
    shift
    local message="$@"

    case $level in
        "INFO")  echo -e "${BLUE}[INFO]${NC}  $message" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC}  $message" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} $message" ;;
        "SUCCESS") echo -e "${GREEN}[✓]${NC} $message" ;;
        "FAIL") echo -e "${RED}[✗]${NC} $message" ;;
        "STEP") echo -e "${CYAN}[→]${NC} $message" ;;
    esac
}

# Function to test endpoint
test_endpoint() {
    local name="$1"
    local url="$2"
    local method="${3:-GET}"
    local data="$4"
    local expected_status="${5:-200}"

    echo ""
    log "STEP" "Testing: $name"
    log "INFO" "  URL: $url"
    log "INFO" "  Method: $method"

    if [ -n "$data" ]; then
        log "INFO" "  Data: $data"
    fi

    # Make request
    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "$url" 2>&1)
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" -H "Content-Type: application/json" -d "$data" "$url" 2>&1)
    fi

    # Extract status code
    status_code=$(echo "$response" | tail -1)
    body=$(echo "$response" | head -n -1)

    # Check status
    if [ "$status_code" = "$expected_status" ]; then
        log "SUCCESS" "$name - Status: $status_code"
        if [ ${#body} -lt 500 ]; then
            echo "  Response: $body"
        else
            echo "  Response: ${body:0:500}... (truncated)"
        fi
        return 0
    else
        log "FAIL" "$name - Expected: $expected_status, Got: $status_code"
        echo "  Response: $body"
        return 1
    fi
}

echo "========================================="
echo "vLLM Semantic Router - Deployment Validation"
echo "Namespace: $NAMESPACE"
echo "========================================="
echo ""

# Step 1: Check pods are running
log "INFO" "Step 1: Checking pod status..."
echo ""
oc get pods -n $NAMESPACE

SEMANTIC_ROUTER_READY=$(oc get pods -n $NAMESPACE -l app=semantic-router --no-headers 2>/dev/null | grep "2/2.*Running" | wc -l)
MODEL_A_READY=$(oc get pods -n $NAMESPACE -l app=model-a --no-headers 2>/dev/null | grep "1/1.*Running" | wc -l)
MODEL_B_READY=$(oc get pods -n $NAMESPACE -l app=model-b --no-headers 2>/dev/null | grep "1/1.*Running" | wc -l)

echo ""
if [ "$SEMANTIC_ROUTER_READY" -eq 0 ]; then
    log "FAIL" "Semantic Router pod is not ready (2/2 Running)"
    exit 1
else
    log "SUCCESS" "Semantic Router pod is ready"
fi

if [ "$MODEL_A_READY" -eq 0 ]; then
    log "FAIL" "Model-A pod is not ready (1/1 Running)"
    exit 1
else
    log "SUCCESS" "Model-A pod is ready"
fi

if [ "$MODEL_B_READY" -eq 0 ]; then
    log "FAIL" "Model-B pod is not ready (1/1 Running)"
    exit 1
else
    log "SUCCESS" "Model-B pod is ready"
fi

# Step 2: Get all endpoints
echo ""
log "INFO" "Step 2: Collecting all endpoints..."
echo ""

# Get route URL
ENVOY_ROUTE=$(oc get route envoy-proxy -n $NAMESPACE -o jsonpath='{.spec.host}' 2>/dev/null)

# Get service IPs
ENVOY_SERVICE_IP=$(oc get svc envoy-proxy -n $NAMESPACE -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
ROUTER_SERVICE_IP=$(oc get svc semantic-router -n $NAMESPACE -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
MODEL_A_SERVICE_IP=$(oc get svc model-a -n $NAMESPACE -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
MODEL_B_SERVICE_IP=$(oc get svc model-b -n $NAMESPACE -o jsonpath='{.spec.clusterIP}' 2>/dev/null)

# Get pod IPs
ROUTER_POD=$(oc get pods -n $NAMESPACE -l app=semantic-router -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
ROUTER_POD_IP=$(oc get pod $ROUTER_POD -n $NAMESPACE -o jsonpath='{.status.podIP}' 2>/dev/null)
MODEL_A_POD=$(oc get pods -n $NAMESPACE -l app=model-a -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
MODEL_A_POD_IP=$(oc get pod $MODEL_A_POD -n $NAMESPACE -o jsonpath='{.status.podIP}' 2>/dev/null)
MODEL_B_POD=$(oc get pods -n $NAMESPACE -l app=model-b -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
MODEL_B_POD_IP=$(oc get pod $MODEL_B_POD -n $NAMESPACE -o jsonpath='{.status.podIP}' 2>/dev/null)

echo "=== External Endpoints ==="
echo "Envoy Route (External):        http://$ENVOY_ROUTE"
echo ""
echo "=== Service Endpoints (ClusterIP) ==="
echo "Envoy Service:                 $ENVOY_SERVICE_IP:8801 (Envoy HTTP)"
echo "Semantic Router Service:       $ROUTER_SERVICE_IP:50051 (gRPC), :8080 (API), :9190 (Metrics)"
echo "Model-A Service:               $MODEL_A_SERVICE_IP:8000"
echo "Model-B Service:               $MODEL_B_SERVICE_IP:8000"
echo ""
echo "=== Pod Endpoints (Direct) ==="
echo "Semantic Router Pod:           $ROUTER_POD_IP"
echo "  - gRPC (ExtProc):            $ROUTER_POD_IP:50051"
echo "  - Classification API:        $ROUTER_POD_IP:8080"
echo "  - Metrics:                   $ROUTER_POD_IP:9190"
echo "  - Envoy Proxy:               $ROUTER_POD_IP:8801"
echo "  - Envoy Admin:               $ROUTER_POD_IP:19000"
echo "Model-A Pod:                   $MODEL_A_POD_IP:8000"
echo "Model-B Pod:                   $MODEL_B_POD_IP:8000"
echo ""

# Step 3: Test endpoints
echo "========================================="
log "INFO" "Step 3: Testing Endpoints"
echo "========================================="

TOTAL_TESTS=0
PASSED_TESTS=0

# Test 3.1: Envoy Admin
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if oc exec -n $NAMESPACE $ROUTER_POD -c envoy-proxy -- curl -s http://localhost:19000/ready | grep -q "LIVE"; then
    log "SUCCESS" "Envoy Admin (localhost:19000/ready)"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    log "FAIL" "Envoy Admin (localhost:19000/ready)"
fi

# Test 3.2: Semantic Router gRPC health
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if oc exec -n $NAMESPACE $ROUTER_POD -c semantic-router -- timeout 2 bash -c "echo '' | nc -z localhost 50051" 2>/dev/null; then
    log "SUCCESS" "Semantic Router gRPC (localhost:50051) - Port Open"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    log "FAIL" "Semantic Router gRPC (localhost:50051) - Port Closed"
fi

# Test 3.3: Semantic Router Metrics
TOTAL_TESTS=$((TOTAL_TESTS + 1))
if oc exec -n $NAMESPACE $ROUTER_POD -c semantic-router -- curl -s http://localhost:9190/metrics | grep -q "^# HELP"; then
    log "SUCCESS" "Semantic Router Metrics (localhost:9190/metrics)"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    log "FAIL" "Semantic Router Metrics (localhost:9190/metrics)"
fi

# Test 3.4: Semantic Router Classification API Health
TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo ""
log "STEP" "Testing Semantic Router Classification API"
CLASSIFY_HEALTH=$(oc exec -n $NAMESPACE $ROUTER_POD -c semantic-router -- curl -s http://localhost:8080/health 2>/dev/null)
if echo "$CLASSIFY_HEALTH" | grep -q "ok\|healthy\|UP"; then
    log "SUCCESS" "Classification API Health (localhost:8080/health)"
    echo "  Response: $CLASSIFY_HEALTH"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    log "FAIL" "Classification API Health (localhost:8080/health)"
    echo "  Response: $CLASSIFY_HEALTH"
fi

# Test 3.5: Model-A Health
TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo ""
log "STEP" "Testing Model-A"
MODEL_A_HEALTH=$(oc exec -n $NAMESPACE $MODEL_A_POD -- curl -s http://localhost:8000/health 2>/dev/null || echo "failed")
if echo "$MODEL_A_HEALTH" | grep -q "ok\|healthy"; then
    log "SUCCESS" "Model-A Health (localhost:8000/health)"
    echo "  Response: $MODEL_A_HEALTH"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    log "FAIL" "Model-A Health (localhost:8000/health)"
    echo "  Response: $MODEL_A_HEALTH"
fi

# Test 3.6: Model-B Health
TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo ""
log "STEP" "Testing Model-B"
MODEL_B_HEALTH=$(oc exec -n $NAMESPACE $MODEL_B_POD -- curl -s http://localhost:8000/health 2>/dev/null || echo "failed")
if echo "$MODEL_B_HEALTH" | grep -q "ok\|healthy"; then
    log "SUCCESS" "Model-B Health (localhost:8000/health)"
    echo "  Response: $MODEL_B_HEALTH"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    log "FAIL" "Model-B Health (localhost:8000/health)"
    echo "  Response: $MODEL_B_HEALTH"
fi

# Test 3.7: Model-A via Service
TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo ""
log "STEP" "Testing Model-A via Service ClusterIP"
MODEL_A_SVC_TEST=$(oc exec -n $NAMESPACE $ROUTER_POD -c semantic-router -- curl -s http://$MODEL_A_SERVICE_IP:8000/health 2>/dev/null || echo "failed")
if echo "$MODEL_A_SVC_TEST" | grep -q "ok\|healthy"; then
    log "SUCCESS" "Model-A via Service ($MODEL_A_SERVICE_IP:8000/health)"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    log "FAIL" "Model-A via Service ($MODEL_A_SERVICE_IP:8000/health)"
fi

# Test 3.8: Model-B via Service
TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo ""
log "STEP" "Testing Model-B via Service ClusterIP"
MODEL_B_SVC_TEST=$(oc exec -n $NAMESPACE $ROUTER_POD -c semantic-router -- curl -s http://$MODEL_B_SERVICE_IP:8000/health 2>/dev/null || echo "failed")
if echo "$MODEL_B_SVC_TEST" | grep -q "ok\|healthy"; then
    log "SUCCESS" "Model-B via Service ($MODEL_B_SERVICE_IP:8000/health)"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    log "FAIL" "Model-B via Service ($MODEL_B_SERVICE_IP:8000/health)"
fi

# Step 4: Test complete request flow
echo ""
echo "========================================="
log "INFO" "Step 4: Testing Complete Request Flow"
echo "========================================="

# Test 4.1: External route (if available)
if [ -n "$ENVOY_ROUTE" ]; then
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo ""
    log "STEP" "Testing External Route"
    log "INFO" "Flow: External Client → Route ($ENVOY_ROUTE) → Envoy Proxy → ExtProc (Router) → Model"

    TEST_QUERY='{"model": "auto", "messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 50}'

    ROUTE_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST "http://$ENVOY_ROUTE/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$TEST_QUERY" 2>&1 || echo -e "\nfailed")

    ROUTE_STATUS=$(echo "$ROUTE_RESPONSE" | tail -1)
    ROUTE_BODY=$(echo "$ROUTE_RESPONSE" | head -n -1)

    if [ "$ROUTE_STATUS" = "200" ]; then
        log "SUCCESS" "External Route - Status: 200"
        echo "  Query: What is 2+2?"
        echo "  Response: ${ROUTE_BODY:0:200}..."
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        log "FAIL" "External Route - Status: $ROUTE_STATUS"
        echo "  Response: $ROUTE_BODY"
    fi
else
    log "WARN" "No external route found. Create one with: oc expose svc/envoy-proxy -n $NAMESPACE"
fi

# Test 4.2: Envoy Proxy via Service (from inside cluster)
TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo ""
log "STEP" "Testing Envoy via Service ClusterIP"
log "INFO" "Flow: Test Pod → Envoy Service ($ENVOY_SERVICE_IP:8801) → ExtProc → Model"

TEST_QUERY='{"model": "auto", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 20}'

ENVOY_SVC_RESPONSE=$(oc exec -n $NAMESPACE $ROUTER_POD -c semantic-router -- \
    curl -s -w "\n%{http_code}" -X POST "http://$ENVOY_SERVICE_IP:8801/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "$TEST_QUERY" 2>&1 || echo -e "\nfailed")

ENVOY_SVC_STATUS=$(echo "$ENVOY_SVC_RESPONSE" | tail -1)
ENVOY_SVC_BODY=$(echo "$ENVOY_SVC_RESPONSE" | head -n -1)

if [ "$ENVOY_SVC_STATUS" = "200" ]; then
    log "SUCCESS" "Envoy via Service - Status: 200"
    echo "  Query: Hello"
    echo "  Response: ${ENVOY_SVC_BODY:0:200}..."
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    log "FAIL" "Envoy via Service - Status: $ENVOY_SVC_STATUS"
    echo "  Response: $ENVOY_SVC_BODY"
fi

# Test 4.3: Classification API
TOTAL_TESTS=$((TOTAL_TESTS + 1))
echo ""
log "STEP" "Testing Classification API (Standalone)"
log "INFO" "Direct classification without routing"

CLASSIFY_QUERY='{"text": "What is the capital of France?"}'

CLASSIFY_RESPONSE=$(oc exec -n $NAMESPACE $ROUTER_POD -c semantic-router -- \
    curl -s -w "\n%{http_code}" -X POST "http://localhost:8080/api/v1/classify/intent" \
    -H "Content-Type: application/json" \
    -d "$CLASSIFY_QUERY" 2>&1 || echo -e "\nfailed")

CLASSIFY_STATUS=$(echo "$CLASSIFY_RESPONSE" | tail -1)
CLASSIFY_BODY=$(echo "$CLASSIFY_RESPONSE" | head -n -1)

if [ "$CLASSIFY_STATUS" = "200" ]; then
    log "SUCCESS" "Classification API - Status: 200"
    echo "  Query: What is the capital of France?"
    echo "  Response: $CLASSIFY_BODY"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    log "FAIL" "Classification API - Status: $CLASSIFY_STATUS"
    echo "  Response: $CLASSIFY_BODY"
fi

# Step 5: Request Flow Visualization
echo ""
echo "========================================="
log "INFO" "Step 5: Request Flow Visualization"
echo "========================================="
echo ""
echo "Complete Request Flow:"
echo ""
echo "  1. External Client"
echo "     ↓ HTTP POST /v1/chat/completions"
echo "  2. OpenShift Route: $ENVOY_ROUTE"
echo "     ↓"
echo "  3. Envoy Service: $ENVOY_SERVICE_IP:8801"
echo "     ↓"
echo "  4. Envoy Proxy Container (in semantic-router pod)"
echo "     ↓ gRPC External Processor call"
echo "  5. Semantic Router Container: $ROUTER_POD_IP:50051"
echo "     ├─ Classify intent"
echo "     ├─ Check cache"
echo "     ├─ PII detection"
echo "     ├─ Jailbreak guard"
echo "     └─ Route decision"
echo "     ↓ Set routing headers"
echo "  6. Back to Envoy Proxy"
echo "     ↓ Route to selected backend"
echo "  7a. Model-A: $MODEL_A_SERVICE_IP:8000 → $MODEL_A_POD_IP:8000"
echo "     OR"
echo "  7b. Model-B: $MODEL_B_SERVICE_IP:8000 → $MODEL_B_POD_IP:8000"
echo "     ↓ LLM inference"
echo "  8. Response back through Envoy"
echo "     ↓"
echo "  9. External Client receives response"
echo ""

# Summary
echo "========================================="
log "INFO" "Validation Summary"
echo "========================================="
echo ""
echo "Total Tests: $TOTAL_TESTS"
echo "Passed: $PASSED_TESTS"
echo "Failed: $((TOTAL_TESTS - PASSED_TESTS))"
echo ""

if [ $PASSED_TESTS -eq $TOTAL_TESTS ]; then
    log "SUCCESS" "All tests passed! ✓"
    echo ""
    log "INFO" "You can test the deployment with:"
    if [ -n "$ENVOY_ROUTE" ]; then
        echo "  curl -X POST http://$ENVOY_ROUTE/v1/chat/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{\"model\": \"auto\", \"messages\": [{\"role\": \"user\", \"content\": \"What is 2+2?\"}]}'"
    else
        echo "  First expose the route: oc expose svc/envoy-proxy -n $NAMESPACE"
    fi
    exit 0
else
    log "ERROR" "Some tests failed. Check the output above for details."
    exit 1
fi
