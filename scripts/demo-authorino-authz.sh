#!/usr/bin/env bash
# Demo: Authorino ext_authz integration with fail-closed credential resolver
# EVERY line of output comes from a real running command. Nothing is pre-printed.
set -eo pipefail

# Resolve repo root relative to this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="$(cd "$SCRIPT_DIR/.." && pwd)"

ENVOY_BIN="$(command -v envoy 2>/dev/null || echo "${FUNC_E_HOME:-$HOME/.local/share/func-e}/envoy-versions/1.37.0/bin/envoy")"
ENVOY_CFG="$WORK/config/testing/envoy-authorino-test.yaml"

G='\033[0;32m'; Y='\033[1;33m'; C='\033[0;36m'; B='\033[1m'; N='\033[0m'
PAUSE="${DEMO_PAUSE:-5}"
banner() { echo -e "\n${B}${C}━━━ $1 ━━━${N}\n"; sleep "$PAUSE"; }

# Print a command, then execute it
run() {
    echo -e "${Y}\$ $*${N}"
    eval "$@"
    echo ""
}

ENVOY_PID=""
PFWD_PID=""
ROUTER_PID=""

cleanup() {
    [ -n "$ENVOY_PID" ] && kill "$ENVOY_PID" 2>/dev/null || true
    [ -n "$PFWD_PID" ] && kill "$PFWD_PID" 2>/dev/null || true
    [ -n "$ROUTER_PID" ] && kill "$ROUTER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ─── Title ───────────────────────────────────────────────────────────────
echo ""
echo -e "${B}${C}╔══════════════════════════════════════════════════════════════════════╗${N}"
echo -e "${B}${C}║  Semantic Router — Authorino ext_authz Token Management Demo        ║${N}"
echo -e "${B}${C}║                                                                      ║${N}"
echo -e "${B}${C}║  • Fail-closed credential resolver (reject if no key found)          ║${N}"
echo -e "${B}${C}║  • Token patterns: 1:1 BYOT, 1:N per-user, N:1 shared               ║${N}"
echo -e "${B}${C}║  • End-to-end: fake keys → OpenAI → 401 proves token forwarding      ║${N}"
echo -e "${B}${C}║  • All output from live Authorino + Envoy + semantic router           ║${N}"
echo -e "${B}${C}╚══════════════════════════════════════════════════════════════════════╝${N}"
echo ""
sleep "$PAUSE"

# ─── 0. Prepare ──────────────────────────────────────────────────────────
banner "Preparing environment"
pkill -f "envoy.*envoy-authorino-test" 2>/dev/null || true
pkill -f "kubectl port-forward.*50055" 2>/dev/null || true
pkill -f "bin/router" 2>/dev/null || true
sleep 2

# ─── 1. Deploy Authorino ────────────────────────────────────────────────
banner "1. Deploy Authorino"
run kubectl apply -f "$WORK/config/authorino/k8s-deploy.yaml"
run kubectl wait --for=condition=ready pod -l app=authorino --timeout=60s
run kubectl get pods -l app=authorino -o wide

# ─── 2. Apply AuthConfig (with localhost hosts for testing) ──────────────
banner "2. AuthConfig policy"

# Write authconfig to a temp file, then apply (avoids strict decoding conflicts on re-apply)
AUTHCFG_TMP="$(mktemp /tmp/authconfig-XXXX.yaml)"
cat > "$AUTHCFG_TMP" <<'AUTHCFG'
apiVersion: authorino.kuadrant.io/v1beta3
kind: AuthConfig
metadata:
  name: semantic-router-auth
spec:
  hosts:
  - "semantic-router.example.com"
  - "localhost"
  - "localhost:8801"
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
AUTHCFG

run cat "$AUTHCFG_TMP"
kubectl delete authconfig semantic-router-auth 2>/dev/null || true
run kubectl apply -f "$AUTHCFG_TMP"
rm -f "$AUTHCFG_TMP"
sleep 2

run kubectl get authconfig semantic-router-auth -o yaml

# ─── 3. Apply token mapping secrets ─────────────────────────────────────
banner "3. Token mapping secrets"

run cat "$WORK/config/authorino/secrets-byot.yaml"
run kubectl apply -f "$WORK/config/authorino/secrets-byot.yaml"

run cat "$WORK/config/authorino/secrets-per-user.yaml"
run kubectl apply -f "$WORK/config/authorino/secrets-per-user.yaml"

run cat "$WORK/config/authorino/secrets-shared.yaml"
run kubectl apply -f "$WORK/config/authorino/secrets-shared.yaml"

run "kubectl get secrets -l app=semantic-router -o custom-columns='NAME:.metadata.name,OPENAI-KEY:.metadata.annotations.openai-key,ANTHROPIC-KEY:.metadata.annotations.anthropic-key'"
sleep 2

# ─── 4. Show Envoy config ───────────────────────────────────────────────
banner "4. Envoy ext_authz config"
run cat "$ENVOY_CFG"

# ─── 4b. Show Router config ─────────────────────────────────────────────
banner "4b. Router config"
run cat "$WORK/config/testing/config.authorino-demo.yaml"

# ─── 5. Start services ──────────────────────────────────────────────────
banner "5. Start services"

# Stop any previous router / envoy
pkill -f "bin/router" 2>/dev/null || true
sleep 1

# Start router with demo config (gpt-4o-mini configured, clear_route_cache enabled)
ROUTER_CFG="$WORK/config/testing/config.authorino-demo.yaml"
export LD_LIBRARY_PATH="${WORK}/candle-binding/target/release:${WORK}/ml-binding/target/release"
"$WORK/bin/router" --config "$ROUTER_CFG" --metrics-port 9190 &>/tmp/router-demo.log &
ROUTER_PID=$!
echo -e "${Y}Started router (PID $ROUTER_PID) with config: $(basename "$ROUTER_CFG")${N}"
sleep 5
run "ss -tlnp | grep 50051"

# Port-forward Authorino
kubectl port-forward svc/authorino 50055:50051 &>/dev/null &
PFWD_PID=$!
sleep 2
run "ss -tlnp | grep 50055"

# Start Envoy
$ENVOY_BIN -c "$ENVOY_CFG" --log-level warn &>/tmp/envoy-demo.log &
ENVOY_PID=$!
sleep 2
run "ss -tlnp | grep 8801"

AUTHORINO_POD=$(kubectl get pods -l app=authorino -o jsonpath='{.items[0].metadata.name}')
LOG_SINCE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

show_log() {
    sleep 1
    echo -e "${C}Authorino log (last request):${N}"
    kubectl logs "$AUTHORINO_POD" --since-time="$LOG_SINCE" 2>/dev/null \
        | grep -iE '"msg"|authconfig|secret|denied|authorized' \
        | tail -8 \
        || echo "(no matching entries)"
    LOG_SINCE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo ""
}

# Helper: run a curl test and show the command + response + Authorino log
test_curl() {
    local label="$1"; shift
    # Print the exact command that will be executed
    echo -e "${Y}\$ curl -s $*${N}"
    # Capture body in a temp file, HTTP status in a variable
    local tmpfile http_code
    tmpfile="$(mktemp)"
    http_code=$(curl -s -o "$tmpfile" -w '%{http_code}' "$@")
    cat "$tmpfile"
    echo ""
    echo -e "${G}HTTP status: ${http_code}${N}"
    rm -f "$tmpfile"
    echo ""
    show_log
}

# ─── 6. Test: No token ──────────────────────────────────────────────────
banner "6. Test: no token"
test_curl "no-token" \
  http://localhost:8801/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello"}]}'

# ─── 7. Test: Invalid token ─────────────────────────────────────────────
banner "7. Test: invalid token"
test_curl "bad-token" \
  http://localhost:8801/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer bad-token-xyz' \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello"}]}'

# ─── 8. Test: BYOT (1:1) — Dave ─────────────────────────────────────────
banner "8. Test: BYOT 1:1 — Dave"
test_curl "dave-byot" \
  http://localhost:8801/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-byot-dave-own-openai-key-abc123' \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Say hi"}],"stream":false}'

# ─── 9. Test: Per-user (1:N) — Alice ────────────────────────────────────
banner "9. Test: per-user 1:N — Alice"
test_curl "alice-per-user" \
  http://localhost:8801/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sr-alice-local-token-7890' \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Say hi"}],"stream":false}'

# ─── 10. Test: Shared (N:1) — Bob ───────────────────────────────────────
banner "10. Test: shared N:1 — Bob"
test_curl "bob-shared" \
  http://localhost:8801/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sr-bob-team-token-1111' \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Say hi"}],"stream":false}'

# ─── 11. Test: Shared (N:1) — Carol ─────────────────────────────────────
banner "11. Test: shared N:1 — Carol"
test_curl "carol-shared" \
  http://localhost:8801/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sr-carol-team-token-2222' \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"Say hi"}],"stream":false}'

# ─── Done ────────────────────────────────────────────────────────────────
banner "Demo complete"
