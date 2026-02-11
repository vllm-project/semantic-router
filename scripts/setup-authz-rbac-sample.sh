#!/usr/bin/env bash
# setup-authz-rbac-demo.sh — Set up RBAC authz demo infrastructure
#
# This script sets up:
#   1. Kubernetes Secrets with authz-groups annotations
#   2. Authorino AuthConfig with RBAC header injection
#   3. Port-forward from Authorino to localhost
#   4. Envoy proxy (Docker)
#   5. Semantic Router (ext_proc)
#
# All ports, image versions, and cluster names are configurable via env vars.
#
# Prerequisites:
#   - Kind cluster with Authorino deployed
#   - Two vLLM instances running
#   - Docker available
#
# Environment variables (all optional, with defaults):
#   KIND_CONTEXT     — Kind context name (default: kind-authorino-test)
#   KUBECONFIG       — Path to kubeconfig (default: ~/.kube/config)
#   ENVOY_PORT       — Envoy listen port (default: 8801)
#   ENVOY_IMAGE      — Envoy Docker image (default: envoyproxy/envoy:v1.31-latest)
#   ROUTER_PORT      — Semantic Router gRPC port (default: 50051)
#   AUTHORINO_PF_PORT — Port-forward local port for Authorino (default: 50052)
#   VLLM_14B_PORT    — 14B model port, for prerequisite check (default: 8000)
#   VLLM_7B_PORT     — 7B model port, for prerequisite check (default: 8001)
#
# Usage:
#   ./scripts/setup-authz-rbac-demo.sh
#   ENVOY_PORT=9090 ROUTER_PORT=50055 ./scripts/setup-authz-rbac-demo.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ─── Configurable parameters ─────────────────────────────────────────
KIND_CONTEXT="${KIND_CONTEXT:-kind-authorino-test}"
KUBECONFIG="${KUBECONFIG:-$HOME/.kube/config}"
ENVOY_PORT="${ENVOY_PORT:-8801}"
ENVOY_IMAGE="${ENVOY_IMAGE:-envoyproxy/envoy:v1.31-latest}"
ROUTER_PORT="${ROUTER_PORT:-50051}"
AUTHORINO_PF_PORT="${AUTHORINO_PF_PORT:-50052}"
VLLM_14B_PORT="${VLLM_14B_PORT:-8000}"
VLLM_7B_PORT="${VLLM_7B_PORT:-8001}"
ENVOY_CONTAINER_NAME="envoy-rbac-test"
ROUTER_CONFIG="config/testing/config.authz-rbac-live.yaml"

echo "================================================================"
echo "  RBAC Authz Demo — Infrastructure Setup"
echo "================================================================"
echo ""
echo "Configuration:"
echo "  KIND_CONTEXT:      ${KIND_CONTEXT}"
echo "  ENVOY_PORT:        ${ENVOY_PORT}"
echo "  ENVOY_IMAGE:       ${ENVOY_IMAGE}"
echo "  ROUTER_PORT:       ${ROUTER_PORT}"
echo "  AUTHORINO_PF_PORT: ${AUTHORINO_PF_PORT}"
echo "  VLLM_14B_PORT:     ${VLLM_14B_PORT}"
echo "  VLLM_7B_PORT:      ${VLLM_7B_PORT}"
echo ""

# ─── Step 1: Verify prerequisites ────────────────────────────────────
echo "[1/6] Checking prerequisites..."

echo -n "  Kind cluster (${KIND_CONTEXT})... "
kubectl --context "${KIND_CONTEXT}" cluster-info >/dev/null 2>&1 \
  && echo "OK" \
  || { echo "FAIL (run: kind create cluster --name authorino-test)"; exit 1; }

echo -n "  Authorino pod... "
kubectl --context "${KIND_CONTEXT}" get pod -l app=authorino -o name 2>/dev/null | grep -q pod/ \
  && echo "OK" \
  || { echo "FAIL (deploy Authorino first)"; exit 1; }

echo -n "  14B vLLM (port ${VLLM_14B_PORT})... "
MODEL_14B=$(curl -sf "http://localhost:${VLLM_14B_PORT}/v1/models" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null) \
  && echo "OK (${MODEL_14B})" \
  || { echo "FAIL (no vLLM on port ${VLLM_14B_PORT})"; exit 1; }

echo -n "  7B vLLM (port ${VLLM_7B_PORT})... "
MODEL_7B=$(curl -sf "http://localhost:${VLLM_7B_PORT}/v1/models" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null) \
  && echo "OK (${MODEL_7B})" \
  || { echo "FAIL (no vLLM on port ${VLLM_7B_PORT})"; exit 1; }

echo ""

# ─── Step 2: Create/update Kubernetes Secrets ─────────────────────────
echo "[2/6] Creating user secrets with authz-groups annotations..."

kubectl --context "${KIND_CONTEXT}" apply -f - <<'EOF'
apiVersion: v1
kind: Secret
metadata:
  name: user-admin
  labels:
    app: semantic-router
    authorino.kuadrant.io/managed-by: authorino
  annotations:
    openai-key: ""
    anthropic-key: ""
    authz-groups: ""
stringData:
  api_key: "sr-admin-token-0000"
type: Opaque
---
apiVersion: v1
kind: Secret
metadata:
  name: user-alice-per-user
  labels:
    app: semantic-router
    authorino.kuadrant.io/managed-by: authorino
  annotations:
    openai-key: "sk-proj-alice-dedicated-openai-key"
    anthropic-key: "sk-ant-alice-dedicated-anthropic-key"
    authz-groups: "engineering"
stringData:
  api_key: "sr-alice-local-token-7890"
type: Opaque
---
apiVersion: v1
kind: Secret
metadata:
  name: user-bob-shared
  labels:
    app: semantic-router
    authorino.kuadrant.io/managed-by: authorino
  annotations:
    openai-key: "sk-proj-shared-team-openai-key-TEAM"
    anthropic-key: "sk-ant-shared-team-anthropic-key-TEAM"
    authz-groups: "premium"
stringData:
  api_key: "sr-bob-team-token-1111"
type: Opaque
---
apiVersion: v1
kind: Secret
metadata:
  name: user-carol-shared
  labels:
    app: semantic-router
    authorino.kuadrant.io/managed-by: authorino
  annotations:
    openai-key: "sk-proj-shared-team-openai-key-TEAM"
    anthropic-key: "sk-ant-shared-team-anthropic-key-TEAM"
    authz-groups: "free"
stringData:
  api_key: "sr-carol-team-token-2222"
type: Opaque
---
apiVersion: v1
kind: Secret
metadata:
  name: user-dave-byot
  labels:
    app: semantic-router
    authorino.kuadrant.io/managed-by: authorino
  annotations:
    openai-key: "sk-byot-dave-own-openai-key-abc123"
    anthropic-key: "sk-ant-byot-dave-own-anthropic-key-xyz"
    authz-groups: "contractor"
stringData:
  api_key: "sk-byot-dave-own-openai-key-abc123"
type: Opaque
EOF

echo "  Created 5 secrets"
echo "  Verifying..."
for s in user-admin user-alice-per-user user-bob-shared user-carol-shared user-dave-byot; do
  groups=$(kubectl --context "${KIND_CONTEXT}" get secret "${s}" \
    -o jsonpath="{.metadata.annotations.authz-groups}" 2>/dev/null)
  echo "    ${s}: authz-groups=${groups:-\"(empty)\"}"
done
echo ""

# ─── Step 3: Apply Authorino AuthConfig ───────────────────────────────
echo "[3/6] Applying Authorino AuthConfig (RBAC header injection)..."

kubectl --context "${KIND_CONTEXT}" apply -f - <<EOF
apiVersion: authorino.kuadrant.io/v1beta3
kind: AuthConfig
metadata:
  name: semantic-router-auth
spec:
  hosts:
  - "semantic-router.example.com"
  - "localhost"
  - "localhost:${ENVOY_PORT}"
  authentication:
    "api-key-users":
      apiKey:
        selector:
          matchLabels:
            app: semantic-router
      credentials:
        authorizationHeader:
          prefix: Bearer
  response:
    success:
      headers:
        x-user-openai-key:
          plain:
            expression: auth.identity.metadata.annotations['openai-key']
        x-user-anthropic-key:
          plain:
            expression: auth.identity.metadata.annotations['anthropic-key']
        x-authz-user-id:
          plain:
            expression: auth.identity.metadata.name
        x-authz-user-groups:
          plain:
            expression: auth.identity.metadata.annotations['authz-groups']
EOF

# Verify
echo -n "  AuthConfig status: "
kubectl --context "${KIND_CONTEXT}" get authconfig semantic-router-auth \
  -o jsonpath='{.status.summary.ready}' 2>/dev/null || echo "unknown"
echo ""
echo ""

# ─── Step 4: Port-forward Authorino ──────────────────────────────────
echo "[4/6] Setting up Authorino port-forward (${AUTHORINO_PF_PORT} → 50051)..."

pkill -f "port-forward svc/authorino ${AUTHORINO_PF_PORT}" 2>/dev/null || true
sleep 1

kubectl --context "${KIND_CONTEXT}" port-forward svc/authorino "${AUTHORINO_PF_PORT}":50051 --address 0.0.0.0 &
PF_PID=$!
sleep 2

if ss -tlnp | grep -q ":${AUTHORINO_PF_PORT}"; then
  echo "  Authorino port-forward OK (PID ${PF_PID})"
else
  echo "  FAIL: port-forward did not start"
  exit 1
fi
echo ""

# ─── Step 5: Start/restart Envoy ─────────────────────────────────────
echo "[5/6] Starting Envoy proxy on port ${ENVOY_PORT}..."

docker rm -f "${ENVOY_CONTAINER_NAME}" 2>/dev/null || true
sleep 1

docker run -d \
  --name "${ENVOY_CONTAINER_NAME}" \
  --network host \
  -v "${REPO_ROOT}/config/envoy.yaml:/etc/envoy/envoy.yaml:ro" \
  "${ENVOY_IMAGE}" \
  -c /etc/envoy/envoy.yaml 2>/dev/null

sleep 2
if ss -tlnp | grep -q ":${ENVOY_PORT}"; then
  echo "  Envoy running on port ${ENVOY_PORT}"
else
  echo "  FAIL: Envoy did not start"
  docker logs "${ENVOY_CONTAINER_NAME}" --tail 5 2>&1
  exit 1
fi
echo ""

# ─── Step 6: Start Semantic Router ───────────────────────────────────
echo "[6/6] Starting Semantic Router on port ${ROUTER_PORT}..."

pkill -f "main.*config.*authz-rbac-live" 2>/dev/null || true
sleep 2

cd "${REPO_ROOT}/src/semantic-router"
go run cmd/main.go \
  --config "../../${ROUTER_CONFIG}" \
  --port "${ROUTER_PORT}" \
  --enable-api=false &
ROUTER_PID=$!
sleep 8

if ss -tlnp | grep -q ":${ROUTER_PORT}"; then
  echo "  Router running on port ${ROUTER_PORT} (PID ${ROUTER_PID})"
else
  echo "  FAIL: Router did not start"
  exit 1
fi
echo ""

# ─── Done ─────────────────────────────────────────────────────────────
echo "================================================================"
echo "  Setup complete."
echo ""
echo "  Models discovered:"
echo "    14B: ${MODEL_14B} (port ${VLLM_14B_PORT})"
echo "    7B:  ${MODEL_7B} (port ${VLLM_7B_PORT})"
echo ""
echo "  Run tests:"
echo "    ENVOY_PORT=${ENVOY_PORT} KIND_CONTEXT=${KIND_CONTEXT} ./scripts/test-authz-rbac-live.sh"
echo "================================================================"
