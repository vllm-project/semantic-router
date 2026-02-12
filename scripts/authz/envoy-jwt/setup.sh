#!/usr/bin/env bash
# setup.sh — Standalone Envoy JWT RBAC setup (no Kubernetes)
#
# Architecture (single Envoy, no K8s):
#   Client (JWT) → Envoy :8802 (jwt_authn + ext_proc + ORIGINAL_DST) → vLLM
#
# Steps:
#   1. Generate RSA keys + JWKS (if not exist)
#   2. Generate JWT tokens for test users
#   3. (Re)start standalone Envoy container with JWKS mounted
#
# Prerequisites:
#   - Docker
#   - Python3 with PyJWT + cryptography
#   - Semantic router running on 127.0.0.1:50053 with EG identity config
#   - vLLM 14B on 127.0.0.1:8000, 7B on 127.0.0.1:8001

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ARTIFACTS_DIR="$SCRIPT_DIR/jwt-artifacts"
ENVOY_CONFIG="$REPO_ROOT/config/authz/envoy-jwt/envoy.yaml"
ENVOY_CONTAINER="envoy-jwt-standalone"
ENVOY_PORT="${ENVOY_PORT:-8802}"
ENVOY_IMAGE="${ENVOY_IMAGE:-envoyproxy/envoy:v1.33-latest}"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Standalone Envoy JWT RBAC Setup (No Kubernetes)        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Envoy port:  $ENVOY_PORT"
echo "  Container:   $ENVOY_CONTAINER"
echo ""

# ── Step 1: Generate RSA keys + JWKS ──
echo "── Step 1: Generate RSA keys + JWKS ──"
if [ -f "$ARTIFACTS_DIR/private-key.pem" ] && [ -f "$ARTIFACTS_DIR/jwks.json" ]; then
    echo "  Keys already exist — skipping (delete $ARTIFACTS_DIR to regenerate)"
else
    python3 "$SCRIPT_DIR/generate-jwt-keys.py" --output-dir "$ARTIFACTS_DIR"
fi
echo ""

# ── Step 2: Generate JWT tokens ──
echo "── Step 2: Generate JWT tokens ──"
python3 "$SCRIPT_DIR/generate-jwt-tokens.py" --output-dir "$ARTIFACTS_DIR"
echo ""

# ── Step 3: Verify prerequisites ──
echo "── Step 3: Verify prerequisites ──"

echo -n "  Semantic router (:50053): "
if ss -tlnp 2>/dev/null | grep -q ':50053'; then
    echo "running"
else
    echo "NOT RUNNING — start router with EG config first"
    exit 1
fi

echo -n "  vLLM 14B (:8000): "
if curl -s --connect-timeout 2 http://127.0.0.1:8000/health -o /dev/null 2>/dev/null; then
    echo "healthy"
else
    echo "NOT REACHABLE"
    exit 1
fi

echo -n "  vLLM 7B (:8001): "
if curl -s --connect-timeout 2 http://127.0.0.1:8001/health -o /dev/null 2>/dev/null; then
    echo "healthy"
else
    echo "NOT REACHABLE"
    exit 1
fi

echo -n "  JWKS file: "
if [ -f "$ARTIFACTS_DIR/jwks.json" ]; then
    echo "$ARTIFACTS_DIR/jwks.json"
else
    echo "MISSING"
    exit 1
fi

echo -n "  Envoy config: "
if [ -f "$ENVOY_CONFIG" ]; then
    echo "$ENVOY_CONFIG"
else
    echo "MISSING"
    exit 1
fi
echo ""

# ── Step 4: (Re)start Envoy ──
echo "── Step 4: (Re)start Envoy container ──"

# Stop existing container if running
if docker ps -q -f name="$ENVOY_CONTAINER" 2>/dev/null | grep -q .; then
    echo "  Stopping existing container..."
    docker rm -f "$ENVOY_CONTAINER" >/dev/null 2>&1
    sleep 1
elif docker ps -aq -f name="$ENVOY_CONTAINER" 2>/dev/null | grep -q .; then
    docker rm -f "$ENVOY_CONTAINER" >/dev/null 2>&1
fi

echo "  Starting Envoy with JWT authentication..."
echo "    Config: $ENVOY_CONFIG"
echo "    JWKS:   $ARTIFACTS_DIR/jwks.json → /etc/envoy/jwks.json"
echo "    Port:   $ENVOY_PORT"

docker run -d \
    --name "$ENVOY_CONTAINER" \
    --network host \
    -v "$ENVOY_CONFIG:/etc/envoy/envoy.yaml:ro" \
    -v "$ARTIFACTS_DIR/jwks.json:/etc/envoy/jwks.json:ro" \
    "$ENVOY_IMAGE" \
    envoy -c /etc/envoy/envoy.yaml --base-id 1

sleep 2

# Verify Envoy started
if docker ps -q -f name="$ENVOY_CONTAINER" 2>/dev/null | grep -q .; then
    echo "  Container started: $ENVOY_CONTAINER"
else
    echo "  ERROR: Container failed to start. Logs:"
    docker logs "$ENVOY_CONTAINER" 2>&1 | tail -20
    exit 1
fi

# Verify port is listening
if ss -tlnp 2>/dev/null | grep -q ":$ENVOY_PORT"; then
    echo "  Listening on :$ENVOY_PORT"
else
    echo "  WARNING: Port $ENVOY_PORT not yet listening. Waiting..."
    sleep 3
    if ss -tlnp 2>/dev/null | grep -q ":$ENVOY_PORT"; then
        echo "  Listening on :$ENVOY_PORT"
    else
        echo "  ERROR: Envoy not listening. Logs:"
        docker logs "$ENVOY_CONTAINER" 2>&1 | tail -20
        exit 1
    fi
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Setup complete!                                        ║"
echo "║                                                         ║"
echo "║  Architecture (no Kubernetes):                          ║"
echo "║    Client (JWT) → Envoy :$ENVOY_PORT                       ║"
echo "║      → jwt_authn (RSA256, claim_to_headers)             ║"
echo "║      → ext_proc (semantic router :50053)                ║"
echo "║      → ORIGINAL_DST → vLLM 14B/7B                      ║"
echo "║                                                         ║"
echo "║  JWT tokens: source $ARTIFACTS_DIR/tokens.env"
echo "║                                                         ║"
echo "║  Quick test:                                            ║"
echo "║    source $ARTIFACTS_DIR/tokens.env"
echo "║    curl -H 'Authorization: Bearer \$JWT_ALICE' \\        ║"
echo "║      http://localhost:$ENVOY_PORT/v1/chat/completions ...  ║"
echo "╚══════════════════════════════════════════════════════════╝"
