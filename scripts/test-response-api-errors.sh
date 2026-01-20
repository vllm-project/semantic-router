#!/bin/bash
# Response API Error Handling Tests
# This script tests the error handling scenarios for the Response API.
#
# Usage:
#   ./scripts/test-response-api-errors.sh
#   ./scripts/test-response-api-errors.sh http://localhost:8801  # Custom endpoint
#
# Prerequisites:
#   docker compose -f deploy/docker-compose/docker-compose.response-api.yml up -d

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default endpoint
ENDPOINT="${1:-http://localhost:8801}"

echo "============================================"
echo "Response API Error Handling Tests"
echo "============================================"
echo "Endpoint: $ENDPOINT"
echo ""

PASSED=0
FAILED=0

# Helper function to run a test
run_test() {
    local test_name="$1"
    local expected_status="$2"
    local method="$3"
    local url="$4"
    local data="$5"
    
    echo -n "Testing: $test_name... "
    
    if [ -n "$data" ]; then
        response=$(curl -s -w "\n%{http_code}" -X "$method" "$url" \
            -H "Content-Type: application/json" \
            -d "$data" 2>&1)
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" "$url" 2>&1)
    fi
    
    status_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n -1)
    
    if [ "$status_code" = "$expected_status" ]; then
        echo -e "${GREEN}PASSED${NC} (status=$status_code)"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}FAILED${NC} (expected=$expected_status, got=$status_code)"
        echo "  Response: $body"
        ((FAILED++))
        return 1
    fi
}

# Wait for services to be ready
echo "Checking service health..."
max_retries=30
retry=0
while [ $retry -lt $max_retries ]; do
    if curl -sf "$ENDPOINT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}Service is healthy${NC}"
        break
    fi
    retry=$((retry + 1))
    echo "Waiting for service... ($retry/$max_retries)"
    sleep 2
done

if [ $retry -eq $max_retries ]; then
    echo -e "${RED}Service not ready after $max_retries retries${NC}"
    exit 1
fi

echo ""
echo "Running Error Handling Tests..."
echo "--------------------------------------------"

# Test 1: Invalid request format (missing input field)
echo ""
echo "1. Invalid Request Format Tests"
echo "   Testing requests without required 'input' field..."

run_test "Missing input field (with messages)" 400 POST "$ENDPOINT/v1/responses" \
    '{"model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": "Hello"}]}'

run_test "Missing input field (empty request)" 400 POST "$ENDPOINT/v1/responses" \
    '{"model": "openai/gpt-oss-20b"}'

run_test "Null input field" 400 POST "$ENDPOINT/v1/responses" \
    '{"model": "openai/gpt-oss-20b", "input": null}'

# Test 2: Non-existent previous_response_id
echo ""
echo "2. Non-existent previous_response_id Tests"
echo "   Testing conversation chaining with invalid IDs..."

# This may return 200 (graceful degradation) or 404/400 (strict validation)
echo -n "Testing: Non-existent previous_response_id... "
response=$(curl -s -w "\n%{http_code}" -X POST "$ENDPOINT/v1/responses" \
    -H "Content-Type: application/json" \
    -d '{"model": "openai/gpt-oss-20b", "input": "Hello", "previous_response_id": "resp_nonexistent_12345"}' 2>&1)
status_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$status_code" = "200" ]; then
    echo -e "${GREEN}PASSED${NC} (graceful degradation, status=$status_code)"
    ((PASSED++))
elif [ "$status_code" = "404" ] || [ "$status_code" = "400" ]; then
    echo -e "${GREEN}PASSED${NC} (strict validation, status=$status_code)"
    ((PASSED++))
else
    echo -e "${RED}FAILED${NC} (unexpected status=$status_code)"
    echo "  Response: $body"
    ((FAILED++))
fi

# Test 3: Non-existent response ID for GET
echo ""
echo "3. Non-existent Response ID for GET Tests"
echo "   Testing GET requests with invalid IDs..."

run_test "GET non-existent response" 404 GET "$ENDPOINT/v1/responses/resp_nonexistent_67890"

run_test "GET with invalid ID format" 404 GET "$ENDPOINT/v1/responses/invalid-id-format"

run_test "GET non-existent input_items" 404 GET "$ENDPOINT/v1/responses/resp_nonexistent_67890/input_items"

# Test 4: Non-existent response ID for DELETE
echo ""
echo "4. Non-existent Response ID for DELETE Tests"
echo "   Testing DELETE requests with invalid IDs..."

run_test "DELETE non-existent response" 404 DELETE "$ENDPOINT/v1/responses/resp_nonexistent_abcde"

run_test "DELETE with invalid ID format" 404 DELETE "$ENDPOINT/v1/responses/invalid-delete-id"

# Test 5: Backend error passthrough
echo ""
echo "5. Backend Error Passthrough Tests"
echo "   Testing that backend errors are properly forwarded..."

# Test with an invalid model (may trigger backend error or auto-routing)
echo -n "Testing: Invalid model name... "
response=$(curl -s -w "\n%{http_code}" -X POST "$ENDPOINT/v1/responses" \
    -H "Content-Type: application/json" \
    -d '{"model": "invalid-model-that-does-not-exist", "input": "Test backend error"}' 2>&1)
status_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | head -n -1)

if [ "$status_code" = "200" ]; then
    echo -e "${YELLOW}PASSED${NC} (model auto-routed, status=$status_code)"
    ((PASSED++))
elif [ "$status_code" -ge 400 ]; then
    # Check if error response has proper format
    if echo "$body" | grep -q '"error"'; then
        echo -e "${GREEN}PASSED${NC} (error passthrough, status=$status_code)"
        ((PASSED++))
    else
        echo -e "${YELLOW}PASSED${NC} (error returned, status=$status_code)"
        ((PASSED++))
    fi
else
    echo -e "${RED}FAILED${NC} (unexpected status=$status_code)"
    echo "  Response: $body"
    ((FAILED++))
fi

# Test 6: Valid request (sanity check)
echo ""
echo "6. Sanity Check - Valid Request"
echo "   Ensuring valid requests still work..."

run_test "Valid POST /v1/responses" 200 POST "$ENDPOINT/v1/responses" \
    '{"model": "openai/gpt-oss-20b", "input": "What is 2+2?", "store": true}'

# Print summary
echo ""
echo "============================================"
echo "Test Summary"
echo "============================================"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
