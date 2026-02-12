#!/usr/bin/env bash
# test-authz-authorino.sh — Live Authorino RBAC integration test
#
# Tests the full pipeline: Client → Envoy (ext_authz) → Authorino → Router → vLLM
#
# Prerequisites:
#   - Kind cluster with Authorino running
#   - kubectl port-forward to Authorino on $AUTHORINO_PF_PORT
#   - Envoy on $ENVOY_PORT with ext_authz + ext_proc
#   - Router on port 50051 with config.authz-rbac-sample.yaml
#   - vLLM 14B on $VLLM_14B_PORT, vLLM 7B on $VLLM_7B_PORT
#
# All values are discovered dynamically from live systems.

set -euo pipefail

# ── Configurable ports ──
ENVOY_PORT="${ENVOY_PORT:-8801}"
VLLM_14B_PORT="${VLLM_14B_PORT:-8000}"
VLLM_7B_PORT="${VLLM_7B_PORT:-8001}"
KIND_CONTEXT="${KIND_CONTEXT:-kind-authorino-test}"

PASS=0
FAIL=0
TOTAL=0

# ── Helper: send request and extract results ──
send_request() {
    local test_name="$1"
    local token="$2"
    local prompt="$3"
    local expect_model_pattern="$4"   # regex for expected model in response
    local expect_http="$5"            # expected HTTP status code

    TOTAL=$((TOTAL + 1))
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "TEST $TOTAL: $test_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Prompt:   $prompt"
    echo "  Expected: HTTP $expect_http, model matching /$expect_model_pattern/"

    local tmpfile
    tmpfile=$(mktemp)

    local http_code
    http_code=$(curl -s -o "$tmpfile" -w '%{http_code}' \
        --max-time 120 \
        "http://127.0.0.1:${ENVOY_PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${token}" \
        -d "{
            \"model\": \"auto\",
            \"messages\": [{\"role\": \"user\", \"content\": \"${prompt}\"}],
            \"max_tokens\": 50
        }" 2>/dev/null) || http_code="000"

    echo "  HTTP:     $http_code"

    # Check HTTP code
    if [ "$http_code" != "$expect_http" ]; then
        echo "  RESULT:   FAIL — expected HTTP $expect_http, got $http_code"
        echo "  Body:     $(head -c 300 "$tmpfile")"
        FAIL=$((FAIL + 1))
        rm -f "$tmpfile"
        return
    fi

    # For non-200, just verify the status code
    if [ "$expect_http" != "200" ]; then
        echo "  Body:     $(head -c 200 "$tmpfile")"
        echo "  RESULT:   PASS"
        PASS=$((PASS + 1))
        rm -f "$tmpfile"
        return
    fi

    # Extract model from response
    local resp_model
    resp_model=$(python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('model',''))" < "$tmpfile" 2>/dev/null) || resp_model=""
    echo "  Model:    $resp_model"

    # Extract a snippet of the response content
    local snippet
    snippet=$(python3 -c "import json,sys; d=json.load(sys.stdin); c=d.get('choices',[{}])[0].get('message',{}).get('content',''); print(c[:120])" < "$tmpfile" 2>/dev/null) || snippet=""
    echo "  Response: ${snippet}..."

    # Check model matches pattern
    if echo "$resp_model" | grep -qE "$expect_model_pattern"; then
        echo "  RESULT:   PASS"
        PASS=$((PASS + 1))
    else
        echo "  RESULT:   FAIL — model '$resp_model' does not match /$expect_model_pattern/"
        FAIL=$((FAIL + 1))
    fi

    rm -f "$tmpfile"
}

# ── Step 1: Discover model names from live vLLM ──
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Authorino RBAC Integration Test                        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "── Step 1: Discover models from live vLLM endpoints ──"

MODEL_14B=$(curl -s "http://127.0.0.1:${VLLM_14B_PORT}/v1/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
MODEL_7B=$(curl -s "http://127.0.0.1:${VLLM_7B_PORT}/v1/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
echo "  14B model: $MODEL_14B (port $VLLM_14B_PORT)"
echo "  7B model:  $MODEL_7B (port $VLLM_7B_PORT)"

# ── Step 2: Discover tokens from K8s secrets ──
echo ""
echo "── Step 2: Discover tokens from K8s secrets ──"

get_token() {
    local secret_name="$1"
    kubectl --context "$KIND_CONTEXT" get secret "$secret_name" \
        -o jsonpath='{.data.api_key}' 2>/dev/null | base64 -d 2>/dev/null
}

get_groups() {
    local secret_name="$1"
    kubectl --context "$KIND_CONTEXT" get secret "$secret_name" \
        -o jsonpath='{.metadata.annotations.authz-groups}' 2>/dev/null
}

TOKEN_ADMIN=$(get_token "user-admin")
TOKEN_ALICE=$(get_token "user-alice-per-user")
TOKEN_BOB=$(get_token "user-bob-shared")
TOKEN_CAROL=$(get_token "user-carol-shared")
TOKEN_DAVE=$(get_token "user-dave-byot")

GROUPS_ADMIN=$(get_groups "user-admin")
GROUPS_ALICE=$(get_groups "user-alice-per-user")
GROUPS_BOB=$(get_groups "user-bob-shared")
GROUPS_CAROL=$(get_groups "user-carol-shared")
GROUPS_DAVE=$(get_groups "user-dave-byot")

echo "  user-admin:          token=${TOKEN_ADMIN:0:12}..., groups='${GROUPS_ADMIN}'"
echo "  user-alice-per-user: token=${TOKEN_ALICE:0:12}..., groups='${GROUPS_ALICE}'"
echo "  user-bob-shared:     token=${TOKEN_BOB:0:12}..., groups='${GROUPS_BOB}'"
echo "  user-carol-shared:   token=${TOKEN_CAROL:0:12}..., groups='${GROUPS_CAROL}'"
echo "  user-dave-byot:      token=${TOKEN_DAVE:0:12}..., groups='${GROUPS_DAVE}'"

# ── Step 3: Verify infra is healthy ──
echo ""
echo "── Step 3: Verify infrastructure ──"

echo -n "  Envoy (port $ENVOY_PORT): "
curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${ENVOY_PORT}/" --max-time 3 2>/dev/null || echo -n "unreachable"
echo ""

echo -n "  vLLM 14B (port $VLLM_14B_PORT): "
curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_14B_PORT}/health" --max-time 3 2>/dev/null || echo -n "unreachable"
echo ""

echo -n "  vLLM 7B (port $VLLM_7B_PORT): "
curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_7B_PORT}/health" --max-time 3 2>/dev/null || echo -n "unreachable"
echo ""

# ── Step 4: Run tests ──
echo ""
echo "── Step 4: Live request tests ──"

# Test 1: Admin user → should get 14B (admin role, priority 300)
send_request \
    "Admin user (user-admin) → expects 14B" \
    "$TOKEN_ADMIN" \
    "What is 2+2?" \
    "$MODEL_14B" \
    "200"

# Test 2: Alice (engineering group, pro_tier) → should get 7B
send_request \
    "Alice (engineering/pro_tier) → expects 7B" \
    "$TOKEN_ALICE" \
    "Hello, how are you?" \
    "$MODEL_7B" \
    "200"

# Test 3: Bob (premium group) + complex query → should get 14B
send_request \
    "Bob (premium) + complex query → expects 14B" \
    "$TOKEN_BOB" \
    "Analyze the trade-offs between microservices and monolithic architecture. Think step by step." \
    "$MODEL_14B" \
    "200"

# Test 4: Bob (premium group) + simple query → should get 7B
send_request \
    "Bob (premium) + simple query → expects 7B" \
    "$TOKEN_BOB" \
    "Hi there" \
    "$MODEL_7B" \
    "200"

# Test 5: Carol (free group) → should get 7B
send_request \
    "Carol (free/free_tier) → expects 7B" \
    "$TOKEN_CAROL" \
    "What is the weather?" \
    "$MODEL_7B" \
    "200"

# Test 6: Dave (contractor group, free_tier) → should get 7B
send_request \
    "Dave (contractor/free_tier) → expects 7B" \
    "$TOKEN_DAVE" \
    "Tell me a joke" \
    "$MODEL_7B" \
    "200"

# Test 7: Invalid token → expects 401 (Authorino rejects)
send_request \
    "Invalid token → expects 401" \
    "invalid-token-does-not-exist" \
    "This should fail" \
    ".*" \
    "401"

# Test 8: No token → expects 401
TOTAL=$((TOTAL + 1))
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST $TOTAL: No token → expects 401"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
no_token_code=$(curl -s -o /dev/null -w '%{http_code}' \
    --max-time 10 \
    "http://127.0.0.1:${ENVOY_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":[{"role":"user","content":"no token"}],"max_tokens":10}' 2>/dev/null) || no_token_code="000"
echo "  HTTP:     $no_token_code"
if [ "$no_token_code" = "401" ] || [ "$no_token_code" = "403" ]; then
    echo "  RESULT:   PASS"
    PASS=$((PASS + 1))
else
    echo "  RESULT:   FAIL — expected 401/403, got $no_token_code"
    FAIL=$((FAIL + 1))
fi

# ── Summary ──
echo ""
echo "══════════════════════════════════════════════════════════"
echo "RESULTS: $PASS passed, $FAIL failed, $TOTAL total"
echo "══════════════════════════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
