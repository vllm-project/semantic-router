#!/usr/bin/env bash
# test-authz-rbac-live.sh — Run RBAC authz signal tests against live infrastructure
#
# All values are read dynamically from the running system:
#   - Tokens from Kubernetes Secrets
#   - Model names from live vLLM /v1/models endpoints
#   - Infrastructure ports from environment variables
#
# Prerequisites (run setup-authz-rbac-demo.sh first):
#   - Kind cluster with Authorino running
#   - Envoy proxy
#   - Semantic Router (ext_proc)
#   - Two vLLM instances (14B on VLLM_14B_PORT, 7B on VLLM_7B_PORT)
#
# Environment variables (all optional, with defaults):
#   ENVOY_PORT       — Envoy listen port (default: 8801)
#   VLLM_14B_PORT    — 14B model port (default: 8000)
#   VLLM_7B_PORT     — 7B model port (default: 8001)
#   KIND_CONTEXT     — Kind context name (default: kind-authorino-test)
#   KUBECONFIG       — Path to kubeconfig (default: ~/.kube/config)
#
# Usage:
#   ./scripts/test-authz-rbac-live.sh
#   ENVOY_PORT=9090 ./scripts/test-authz-rbac-live.sh
#
set -euo pipefail

# ─── Configurable parameters (from env, with defaults) ───────────────
ENVOY_PORT="${ENVOY_PORT:-8801}"
VLLM_14B_PORT="${VLLM_14B_PORT:-8000}"
VLLM_7B_PORT="${VLLM_7B_PORT:-8001}"
KIND_CONTEXT="${KIND_CONTEXT:-kind-authorino-test}"
ENVOY_URL="http://localhost:${ENVOY_PORT}"

PASS=0
FAIL=0

# ─── Dynamic discovery ───────────────────────────────────────────────
echo "================================================================"
echo "  RBAC Authz Signal — Live Integration Tests"
echo "================================================================"
echo ""

# Discover model names from live vLLM endpoints
echo "Discovering models from live vLLM endpoints..."
MODEL_14B=$(curl -sf "http://localhost:${VLLM_14B_PORT}/v1/models" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null) \
  || { echo "FAIL: Cannot reach vLLM on port ${VLLM_14B_PORT}"; exit 1; }

MODEL_7B=$(curl -sf "http://localhost:${VLLM_7B_PORT}/v1/models" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null) \
  || { echo "FAIL: Cannot reach vLLM on port ${VLLM_7B_PORT}"; exit 1; }

echo "  14B: ${MODEL_14B} (port ${VLLM_14B_PORT})"
echo "  7B:  ${MODEL_7B} (port ${VLLM_7B_PORT})"
echo ""

# Read tokens from Kubernetes Secrets (not hardcoded)
echo "Reading tokens from Kubernetes Secrets..."
read_token() {
  local secret_name="$1"
  local token
  token=$(kubectl --context "${KIND_CONTEXT}" get secret "${secret_name}" \
    -o jsonpath='{.data.api_key}' 2>/dev/null | base64 -d 2>/dev/null) \
    || { echo "FAIL: Cannot read secret ${secret_name}"; exit 1; }
  echo "${token}"
}

read_groups() {
  local secret_name="$1"
  kubectl --context "${KIND_CONTEXT}" get secret "${secret_name}" \
    -o jsonpath="{.metadata.annotations.authz-groups}" 2>/dev/null || echo "(none)"
}

TOKEN_ADMIN=$(read_token "user-admin")
TOKEN_ALICE=$(read_token "user-alice-per-user")
TOKEN_BOB=$(read_token "user-bob-shared")
TOKEN_CAROL=$(read_token "user-carol-shared")
TOKEN_DAVE=$(read_token "user-dave-byot")

GROUPS_ADMIN=$(read_groups "user-admin")
GROUPS_ALICE=$(read_groups "user-alice-per-user")
GROUPS_BOB=$(read_groups "user-bob-shared")
GROUPS_CAROL=$(read_groups "user-carol-shared")
GROUPS_DAVE=$(read_groups "user-dave-byot")

echo ""
echo "Infrastructure:"
echo "  Envoy:     ${ENVOY_URL}"
echo "  14B model: localhost:${VLLM_14B_PORT} (${MODEL_14B})"
echo "  7B model:  localhost:${VLLM_7B_PORT} (${MODEL_7B})"
echo ""
echo "Users (from K8s Secrets):"
echo "  admin  (user-admin)          groups: ${GROUPS_ADMIN:-"(none)"}"
echo "  alice  (user-alice-per-user) groups: ${GROUPS_ALICE}"
echo "  bob    (user-bob-shared)     groups: ${GROUPS_BOB}"
echo "  carol  (user-carol-shared)   groups: ${GROUPS_CAROL}"
echo "  dave   (user-dave-byot)      groups: ${GROUPS_DAVE}"
echo ""
echo "----------------------------------------------------------------"
echo ""

# ─── Helpers ──────────────────────────────────────────────────────────
send_request() {
  local token="$1"
  local message="$2"
  local max_tokens="${3:-30}"

  curl -sf "${ENVOY_URL}/v1/chat/completions" \
    -H "Authorization: Bearer ${token}" \
    -H "Content-Type: application/json" \
    -H "Host: localhost:${ENVOY_PORT}" \
    -d "{
      \"model\": \"auto\",
      \"messages\": [{\"role\": \"user\", \"content\": \"${message}\"}],
      \"max_tokens\": ${max_tokens}
    }" 2>&1
}

send_request_status() {
  local token="$1"
  local message="$2"

  curl -s -o /dev/null -w "%{http_code}" "${ENVOY_URL}/v1/chat/completions" \
    -H "Authorization: Bearer ${token}" \
    -H "Content-Type: application/json" \
    -H "Host: localhost:${ENVOY_PORT}" \
    -d "{
      \"model\": \"auto\",
      \"messages\": [{\"role\": \"user\", \"content\": \"${message}\"}],
      \"max_tokens\": 10
    }" 2>&1
}

extract_model() {
  python3 -c "import sys,json; print(json.load(sys.stdin)['model'])" 2>/dev/null || echo "PARSE_ERROR"
}

check_model() {
  local test_name="$1"
  local response="$2"
  local expected_model="$3"

  actual_model=$(echo "${response}" | extract_model)

  if [[ "${actual_model}" == "${expected_model}" ]]; then
    echo "  PASS  ${test_name}"
    echo "        → model: ${actual_model}"
    PASS=$((PASS + 1))
  else
    echo "  FAIL  ${test_name}"
    echo "        expected: ${expected_model}"
    echo "        actual:   ${actual_model}"
    FAIL=$((FAIL + 1))
  fi
}

check_status() {
  local test_name="$1"
  local actual_status="$2"
  local expected_status="$3"

  if [[ "${actual_status}" == "${expected_status}" ]]; then
    echo "  PASS  ${test_name}"
    echo "        → HTTP ${actual_status}"
    PASS=$((PASS + 1))
  else
    echo "  FAIL  ${test_name}"
    echo "        expected: HTTP ${expected_status}"
    echo "        actual:   HTTP ${actual_status}"
    FAIL=$((FAIL + 1))
  fi
}

# ─── Test 1: Admin → 14B ─────────────────────────────────────────────
echo "Test 1: Admin user, simple query → should route to 14B"
resp=$(send_request "${TOKEN_ADMIN}" "What is 2+2?")
check_model "admin simple query" "${resp}" "${MODEL_14B}"
echo ""

# ─── Test 2: Alice (engineering/pro_tier) → 7B ───────────────────────
echo "Test 2: Alice (${GROUPS_ALICE}/pro_tier), simple query → should route to 7B"
resp=$(send_request "${TOKEN_ALICE}" "What is 2+2?")
check_model "alice simple query" "${resp}" "${MODEL_7B}"
echo ""

# ─── Test 3: Bob (premium) + complex query → 14B ─────────────────────
echo "Test 3: Bob (${GROUPS_BOB}), complex query → should route to 14B"
resp=$(send_request "${TOKEN_BOB}" "Please analyze and explain why the sky is blue, think step by step" 50)
check_model "bob complex query" "${resp}" "${MODEL_14B}"
echo ""

# ─── Test 4: Bob (premium) + simple query → 7B ───────────────────────
echo "Test 4: Bob (${GROUPS_BOB}), simple query → should route to 7B (cost savings)"
resp=$(send_request "${TOKEN_BOB}" "Hi there")
check_model "bob simple query" "${resp}" "${MODEL_7B}"
echo ""

# ─── Test 5: Carol (free) → 7B ───────────────────────────────────────
echo "Test 5: Carol (${GROUPS_CAROL}), simple query → should route to 7B"
resp=$(send_request "${TOKEN_CAROL}" "What is the capital of France?")
check_model "carol simple query" "${resp}" "${MODEL_7B}"
echo ""

# ─── Test 6: Dave (contractor/free_tier) → 7B ────────────────────────
echo "Test 6: Dave (${GROUPS_DAVE}/free_tier), simple query → should route to 7B"
resp=$(send_request "${TOKEN_DAVE}" "Hello world")
check_model "dave simple query" "${resp}" "${MODEL_7B}"
echo ""

# ─── Test 7: Invalid token → 401 ─────────────────────────────────────
echo "Test 7: Invalid token → should return 401"
status=$(send_request_status "fake-invalid-token-xyz" "Hello")
check_status "invalid token" "${status}" "401"
echo ""

# ─── Test 8: No token → 401 ──────────────────────────────────────────
echo "Test 8: No token → should return 401"
status=$(curl -s -o /dev/null -w "%{http_code}" "${ENVOY_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Host: localhost:${ENVOY_PORT}" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Hello"}],"max_tokens":10}' 2>&1)
check_status "no token" "${status}" "401"
echo ""

# ─── Summary ──────────────────────────────────────────────────────────
echo "================================================================"
echo "  Results: ${PASS} passed, ${FAIL} failed (total $((PASS + FAIL)))"
echo "================================================================"

if [[ ${FAIL} -gt 0 ]]; then
  exit 1
fi
