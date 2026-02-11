#!/usr/bin/env bash
# test-authz-eg.sh — Live Envoy Gateway JWT simulation test
#
# Tests the router with custom identity headers (x-jwt-sub, x-jwt-groups)
# that simulate what Envoy Gateway's SecurityPolicy claim_to_headers would inject
# after JWT validation.
#
# Pipeline: Client → Envoy (no ext_authz) → Router (custom identity) → vLLM
#
# Since we don't have EG running, the client sends x-jwt-sub / x-jwt-groups
# headers directly — this is exactly what EG would inject after validating
# the JWT and extracting claims.
#
# Prerequisites:
#   - Envoy on $EG_ENVOY_PORT (8802) with ext_proc to router on 50053, NO ext_authz
#   - Router on 50053 with config.eg-rbac-test.yaml (custom identity headers)
#   - vLLM 14B on $VLLM_14B_PORT, vLLM 7B on $VLLM_7B_PORT

set -euo pipefail

# ── Configurable ports ──
EG_ENVOY_PORT="${EG_ENVOY_PORT:-8802}"
VLLM_14B_PORT="${VLLM_14B_PORT:-8000}"
VLLM_7B_PORT="${VLLM_7B_PORT:-8001}"

PASS=0
FAIL=0
TOTAL=0

# ── Helper: send request with custom identity headers ──
send_eg_request() {
    local test_name="$1"
    local user_id="$2"         # x-jwt-sub (JWT sub claim)
    local user_groups="$3"     # x-jwt-groups (JWT groups claim, comma-separated)
    local prompt="$4"
    local expect_model_pattern="$5"
    local expect_http="$6"

    TOTAL=$((TOTAL + 1))
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "TEST $TOTAL: $test_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  x-jwt-sub:    $user_id"
    echo "  x-jwt-groups: $user_groups"
    echo "  Prompt:        $prompt"
    echo "  Expected:      HTTP $expect_http, model matching /$expect_model_pattern/"

    local tmpfile
    tmpfile=$(mktemp)

    # Build curl command with identity headers
    # These headers simulate what EG claim_to_headers would inject from the JWT
    local curl_args=(
        -s -o "$tmpfile" -w '%{http_code}'
        --max-time 120
        "http://127.0.0.1:${EG_ENVOY_PORT}/v1/chat/completions"
        -H "Content-Type: application/json"
    )

    # Add identity headers (simulating EG claim_to_headers injection)
    if [ -n "$user_id" ]; then
        curl_args+=(-H "x-jwt-sub: ${user_id}")
    fi
    if [ -n "$user_groups" ]; then
        curl_args+=(-H "x-jwt-groups: ${user_groups}")
    fi

    curl_args+=(-d "{
        \"model\": \"auto\",
        \"messages\": [{\"role\": \"user\", \"content\": \"${prompt}\"}],
        \"max_tokens\": 50
    }")

    local http_code
    http_code=$(curl "${curl_args[@]}" 2>/dev/null) || http_code="000"

    echo "  HTTP:          $http_code"

    # Check HTTP code
    if [ "$http_code" != "$expect_http" ]; then
        echo "  RESULT:        FAIL — expected HTTP $expect_http, got $http_code"
        echo "  Body:          $(head -c 300 "$tmpfile")"
        FAIL=$((FAIL + 1))
        rm -f "$tmpfile"
        return
    fi

    # For non-200, just verify status
    if [ "$expect_http" != "200" ]; then
        echo "  Body:          $(head -c 200 "$tmpfile")"
        echo "  RESULT:        PASS"
        PASS=$((PASS + 1))
        rm -f "$tmpfile"
        return
    fi

    # Extract model from response
    local resp_model
    resp_model=$(python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('model',''))" < "$tmpfile" 2>/dev/null) || resp_model=""
    echo "  Model:         $resp_model"

    # Extract response snippet
    local snippet
    snippet=$(python3 -c "import json,sys; d=json.load(sys.stdin); c=d.get('choices',[{}])[0].get('message',{}).get('content',''); print(c[:120])" < "$tmpfile" 2>/dev/null) || snippet=""
    echo "  Response:      ${snippet}..."

    # Check model matches
    if echo "$resp_model" | grep -qE "$expect_model_pattern"; then
        echo "  RESULT:        PASS"
        PASS=$((PASS + 1))
    else
        echo "  RESULT:        FAIL — model '$resp_model' does not match /$expect_model_pattern/"
        FAIL=$((FAIL + 1))
    fi

    rm -f "$tmpfile"
}

# ── Step 1: Discover model names ──
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Envoy Gateway JWT RBAC Integration Test                ║"
echo "║  (Simulated claim_to_headers via x-jwt-sub/x-jwt-groups)║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "── Step 1: Discover models from live vLLM endpoints ──"

MODEL_14B=$(curl -s "http://127.0.0.1:${VLLM_14B_PORT}/v1/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
MODEL_7B=$(curl -s "http://127.0.0.1:${VLLM_7B_PORT}/v1/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
echo "  14B model: $MODEL_14B (port $VLLM_14B_PORT)"
echo "  7B model:  $MODEL_7B (port $VLLM_7B_PORT)"

# ── Step 2: Verify infra ──
echo ""
echo "── Step 2: Verify infrastructure ──"

echo -n "  EG Envoy (port $EG_ENVOY_PORT): "
eg_check=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${EG_ENVOY_PORT}/" --max-time 3 2>/dev/null) || eg_check="unreachable"
echo "$eg_check"

echo -n "  vLLM 14B (port $VLLM_14B_PORT): "
curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_14B_PORT}/health" --max-time 3 2>/dev/null || echo -n "unreachable"
echo ""

echo -n "  vLLM 7B (port $VLLM_7B_PORT): "
curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_7B_PORT}/health" --max-time 3 2>/dev/null || echo -n "unreachable"
echo ""

echo ""
echo "── Step 3: Show router identity config ──"
echo "  Router reads identity from custom headers (simulating EG claim_to_headers):"
echo "    user_id_header:     x-jwt-sub    (JWT 'sub' claim)"
echo "    user_groups_header: x-jwt-groups (JWT 'groups' claim)"
echo ""
echo "  Role bindings:"
echo "    platform-admins → admin        → 14B + reasoning"
echo "    premium-tier    → premium_user → 14B (complex) or 7B (simple)"
echo "    free-tier       → free_user    → 7B only"

# ── Step 3: Run tests ──
echo ""
echo "── Step 4: Live request tests ──"

# Test 1: Admin (platform-admins group) → 14B
send_eg_request \
    "Admin (platform-admins) → expects 14B" \
    "alice" \
    "platform-admins" \
    "What is 2+2?" \
    "$MODEL_14B" \
    "200"

# Test 2: Premium user + complex query → 14B
send_eg_request \
    "Premium (premium-tier) + complex → expects 14B" \
    "bob" \
    "premium-tier" \
    "Analyze the differences between REST and GraphQL. Think step by step." \
    "$MODEL_14B" \
    "200"

# Test 3: Premium user + simple query → 7B
send_eg_request \
    "Premium (premium-tier) + simple → expects 7B" \
    "bob" \
    "premium-tier" \
    "Hi there" \
    "$MODEL_7B" \
    "200"

# Test 4: Free user → 7B
send_eg_request \
    "Free (free-tier) → expects 7B" \
    "carol" \
    "free-tier" \
    "What is the weather?" \
    "$MODEL_7B" \
    "200"

# Test 5: User with multiple groups (premium-tier,platform-admins) → admin (higher priority)
send_eg_request \
    "Multi-group (premium-tier,platform-admins) → expects 14B (admin priority)" \
    "dave" \
    "premium-tier,platform-admins" \
    "Hello" \
    "$MODEL_14B" \
    "200"

# Test 6: Unknown user, no groups → no role match → default model (7B)
send_eg_request \
    "Unknown user, no groups → expects 7B (default)" \
    "unknown-user" \
    "" \
    "Tell me a joke" \
    "$MODEL_7B" \
    "200"

# Test 7: No identity headers at all → authz classifier rejects empty user ID
# The router returns 403 in the response body (error.code=403) with the rejection reason.
# The HTTP transport status may be 200 (ext_proc protocol) or 403 (direct rejection).
TOTAL=$((TOTAL + 1))
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST $TOTAL: No identity headers → expects authz rejection (403 in body)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  x-jwt-sub:    (empty)"
echo "  x-jwt-groups: (empty)"
no_id_tmpfile=$(mktemp)
no_id_http=$(curl -s -o "$no_id_tmpfile" -w '%{http_code}' \
    --max-time 30 \
    "http://127.0.0.1:${EG_ENVOY_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"auto","messages":[{"role":"user","content":"This should fail"}],"max_tokens":10}' 2>/dev/null) || no_id_http="000"
echo "  HTTP:          $no_id_http"

# Check for 403 error code in the response body or HTTP status
no_id_body=$(cat "$no_id_tmpfile")
echo "  Body:          ${no_id_body:0:200}"
if echo "$no_id_body" | python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('error',{}).get('code')==403 else 1)" 2>/dev/null; then
    echo "  RESULT:        PASS (403 in response body — authz correctly rejected)"
    PASS=$((PASS + 1))
elif [ "$no_id_http" = "403" ]; then
    echo "  RESULT:        PASS (HTTP 403 — authz correctly rejected)"
    PASS=$((PASS + 1))
else
    echo "  RESULT:        FAIL — expected 403 rejection, got HTTP $no_id_http without error code"
    FAIL=$((FAIL + 1))
fi
rm -f "$no_id_tmpfile"

# ── Summary ──
echo ""
echo "══════════════════════════════════════════════════════════"
echo "RESULTS: $PASS passed, $FAIL failed, $TOTAL total"
echo "══════════════════════════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
