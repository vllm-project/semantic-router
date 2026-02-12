#!/usr/bin/env bash
# test.sh — Standalone Envoy JWT RBAC integration test (no Kubernetes)
#
# Architecture:
#   Client (JWT) → Envoy :8802 (jwt_authn + ext_proc + ORIGINAL_DST) → vLLM
#
# Sends REAL signed JWTs. Envoy validates the RSA256 signature against
# a local JWKS file, extracts claims to headers, calls the semantic router
# via ext_proc, and routes to the correct vLLM backend.
#
# Prerequisites:
#   - setup.sh has been run (Envoy container on $ENVOY_PORT)
#   - JWT tokens generated in jwt-artifacts/tokens.env
#   - vLLM 14B on $VLLM_14B_PORT, vLLM 7B on $VLLM_7B_PORT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACTS_DIR="$SCRIPT_DIR/jwt-artifacts"

# ── Configurable ports ──
ENVOY_PORT="${ENVOY_PORT:-8802}"
VLLM_14B_PORT="${VLLM_14B_PORT:-8000}"
VLLM_7B_PORT="${VLLM_7B_PORT:-8001}"

PASS=0
FAIL=0
TOTAL=0

# ── Load JWT tokens ──
TOKENS_FILE="$ARTIFACTS_DIR/tokens.env"
if [ ! -f "$TOKENS_FILE" ]; then
    echo "ERROR: JWT tokens not found at $TOKENS_FILE"
    echo "Run setup.sh first."
    exit 1
fi
source "$TOKENS_FILE"

# ── Helper: send request with real JWT ──
send_jwt_request() {
    local test_name="$1"
    local jwt_token="$2"
    local jwt_label="$3"
    local prompt="$4"
    local expect_model_pattern="$5"
    local expect_http="$6"

    TOTAL=$((TOTAL + 1))
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "TEST $TOTAL: $test_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  JWT:       $jwt_label"
    echo "  Prompt:    $prompt"
    echo "  Expected:  HTTP $expect_http, model matching /$expect_model_pattern/"

    local tmpfile
    tmpfile=$(mktemp)

    local curl_args=(
        -s -o "$tmpfile" -w '%{http_code}'
        --max-time 120
        "http://127.0.0.1:${ENVOY_PORT}/v1/chat/completions"
        -H "Content-Type: application/json"
    )

    if [ -n "$jwt_token" ]; then
        curl_args+=(-H "Authorization: Bearer ${jwt_token}")
    fi

    curl_args+=(-d "{
        \"model\": \"auto\",
        \"messages\": [{\"role\": \"user\", \"content\": \"${prompt}\"}],
        \"max_tokens\": 50
    }")

    local http_code
    http_code=$(curl "${curl_args[@]}" 2>/dev/null) || http_code="000"
    echo "  HTTP:      $http_code"

    if [ "$http_code" != "$expect_http" ]; then
        echo "  RESULT:    FAIL — expected HTTP $expect_http, got $http_code"
        local body
        body=$(head -c 300 "$tmpfile")
        echo "  Body:      $body"
        FAIL=$((FAIL + 1))
        rm -f "$tmpfile"
        return
    fi

    if [ "$expect_http" != "200" ]; then
        echo "  RESULT:    PASS"
        PASS=$((PASS + 1))
        rm -f "$tmpfile"
        return
    fi

    local resp_model
    resp_model=$(python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('model',''))" < "$tmpfile" 2>/dev/null) || resp_model=""
    echo "  Model:     $resp_model"

    local snippet
    snippet=$(python3 -c "import json,sys; d=json.load(sys.stdin); c=d.get('choices',[{}])[0].get('message',{}).get('content',''); print(c[:120])" < "$tmpfile" 2>/dev/null) || snippet=""
    echo "  Response:  ${snippet}..."

    if echo "$resp_model" | grep -qE "$expect_model_pattern"; then
        echo "  RESULT:    PASS"
        PASS=$((PASS + 1))
    else
        echo "  RESULT:    FAIL — model '$resp_model' does not match /$expect_model_pattern/"
        FAIL=$((FAIL + 1))
    fi

    rm -f "$tmpfile"
}

# ── Header ──
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Standalone Envoy JWT RBAC — Full Integration Test      ║"
echo "║  (Real JWTs, No Kubernetes)                             ║"
echo "╚══════════════════════════════════════════════════════════╝"

# ── Discover models ──
echo ""
echo "── Step 1: Discover models from live vLLM endpoints ──"
MODEL_14B=$(curl -s "http://127.0.0.1:${VLLM_14B_PORT}/v1/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
MODEL_7B=$(curl -s "http://127.0.0.1:${VLLM_7B_PORT}/v1/models" | python3 -c "import json,sys; print(json.load(sys.stdin)['data'][0]['id'])")
echo "  14B model: $MODEL_14B (port $VLLM_14B_PORT)"
echo "  7B model:  $MODEL_7B (port $VLLM_7B_PORT)"

# ── Verify infra ──
echo ""
echo "── Step 2: Verify infrastructure ──"

echo -n "  Envoy (port $ENVOY_PORT): "
envoy_check=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${ENVOY_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" -d '{}' --max-time 3 2>/dev/null) || envoy_check="unreachable"
echo "$envoy_check (401 = JWT required, working)"

echo -n "  vLLM 14B: "
curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_14B_PORT}/health" --max-time 3 2>/dev/null || echo -n "unreachable"
echo ""

echo -n "  vLLM 7B: "
curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${VLLM_7B_PORT}/health" --max-time 3 2>/dev/null || echo -n "unreachable"
echo ""

# ── Show architecture ──
echo ""
echo "── Step 3: Architecture ──"
echo "  Client (JWT) → Envoy :$ENVOY_PORT"
echo "    → jwt_authn (RSA256, local JWKS, claim_to_headers)"
echo "    → ext_proc (semantic router :50053, RBAC role matching)"
echo "    → ORIGINAL_DST → vLLM 14B (:$VLLM_14B_PORT) / 7B (:$VLLM_7B_PORT)"
echo ""
echo "  Role bindings (from router config):"
echo "    platform-admins → admin        → 14B + reasoning"
echo "    premium-tier    → premium_user → 14B (complex) or 7B (simple)"
echo "    free-tier       → free_user    → 7B only"

# ── Run tests ──
echo ""
echo "── Step 4: Live request tests (real JWTs) ──"

# Test 1: Admin → 14B
send_jwt_request \
    "Alice (admin/platform-admins) → expects 14B" \
    "$JWT_ALICE" "JWT_ALICE (sub=alice, groups=platform-admins)" \
    "What is 2+2?" \
    "$MODEL_14B" "200"

# Test 2: Premium + complex → 14B
send_jwt_request \
    "Bob (premium) + complex query → expects 14B" \
    "$JWT_BOB" "JWT_BOB (sub=bob, groups=premium-tier)" \
    "Analyze the differences between REST and GraphQL. Think step by step." \
    "$MODEL_14B" "200"

# Test 3: Premium + simple → 7B
send_jwt_request \
    "Bob (premium) + simple query → expects 7B" \
    "$JWT_BOB" "JWT_BOB (sub=bob, groups=premium-tier)" \
    "Hi there" \
    "$MODEL_7B" "200"

# Test 4: Free → 7B
send_jwt_request \
    "Carol (free/free-tier) → expects 7B" \
    "$JWT_CAROL" "JWT_CAROL (sub=carol, groups=free-tier)" \
    "What is the weather?" \
    "$MODEL_7B" "200"

# Test 5: Multi-group → admin priority → 14B
send_jwt_request \
    "Dave (premium+admin) → expects 14B (admin priority)" \
    "$JWT_DAVE" "JWT_DAVE (sub=dave, groups=premium-tier,platform-admins)" \
    "Hello" \
    "$MODEL_14B" "200"

# Test 6: Unknown user (no groups) → default model
send_jwt_request \
    "Unknown (no groups) → expects 7B (default)" \
    "$JWT_UNKNOWN" "JWT_UNKNOWN (sub=unknown, no groups)" \
    "Tell me a joke" \
    "$MODEL_7B" "200"

# Test 7: No JWT → 401
send_jwt_request \
    "No JWT → expects 401 (Envoy rejects)" \
    "" "(none)" \
    "This should fail" \
    ".*" "401"

# Test 8: Expired JWT → 401
send_jwt_request \
    "Expired JWT → expects 401 (Envoy rejects)" \
    "$JWT_EXPIRED" "JWT_EXPIRED (sub=expired, exp in past)" \
    "This should fail" \
    ".*" "401"

# ── Summary ──
echo ""
echo "══════════════════════════════════════════════════════════"
echo "RESULTS: $PASS passed, $FAIL failed, $TOTAL total"
echo "══════════════════════════════════════════════════════════"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
