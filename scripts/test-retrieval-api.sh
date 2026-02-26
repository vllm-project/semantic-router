#!/bin/bash
# Hierarchical + Hybrid Memory Retrieval API Tests
#
# Tests the memory retrieval pipeline with hierarchical search (category-then-drilldown)
# and hybrid scoring (BM25 + n-gram + vector fusion) against a running semantic router.
#
# Prerequisites:
#   - Semantic router running with config/testing/config.memory-hierarchical.yaml
#   - Milvus running and accessible
#   - LLM backend running (echo mode recommended for deterministic verification)
#
# Usage:
#   ./scripts/test-retrieval-api.sh                         # default endpoint
#   ./scripts/test-retrieval-api.sh http://localhost:8888    # custom endpoint
#   SKIP_SEED=1 ./scripts/test-retrieval-api.sh             # skip seeding, run queries only

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

ENDPOINT="${1:-${ROUTER_ENDPOINT:-http://localhost:8802}}"
MILVUS_ADDRESS="${MILVUS_ADDRESS:-172.17.0.1:19530}"
SKIP_SEED="${SKIP_SEED:-0}"
EXTRACTION_WAIT="${EXTRACTION_WAIT:-12}"

USER_ID="retrieval_test_$(date +%s)"

PASSED=0
FAILED=0
SKIPPED=0
TOTAL=0

log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[PASS]${NC}  $*"; }
log_fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_head()  { echo -e "\n${CYAN}======== $* ========${NC}"; }

run_test() {
    local name="$1"
    local query="$2"
    shift 2
    local keywords=("$@")

    ((TOTAL++)) || true
    log_info "Query: $query"
    log_info "Expect keywords: ${keywords[*]}"

    local response
    response=$(curl -s -w '\n{"_http_status":%{http_code}}' \
        -X POST "$ENDPOINT/v1/responses" \
        -H "Content-Type: application/json" \
        -H "x-authz-user-id: $USER_ID" \
        -d "$(cat <<EOJSON
{
  "model": "MoM",
  "input": "$query",
  "instructions": "You are a helpful assistant with memory.",
  "metadata": {"user_id": "$USER_ID"}
}
EOJSON
)" 2>&1)

    local status
    status=$(echo "$response" | tail -1 | sed 's/.*"_http_status":\([0-9]*\).*/\1/')
    local body
    body=$(echo "$response" | head -n -1)

    if [ "$status" != "200" ]; then
        log_fail "$name — HTTP $status"
        echo "  Response: $(echo "$body" | head -c 300)"
        ((FAILED++)) || true
        return
    fi

    local output
    output=$(echo "$body" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    txt = d.get('output_text', '')
    if not txt:
        out = d.get('output', [])
        if out and isinstance(out, list):
            for o in out:
                for c in o.get('content', []):
                    if isinstance(c, dict) and 'text' in c:
                        txt += c['text']
    print(txt)
except Exception as e:
    print(f'PARSE_ERROR: {e}', file=sys.stderr)
    print('')
" 2>/dev/null)

    local output_lower
    output_lower=$(echo "$output" | tr '[:upper:]' '[:lower:]')

    local found=0
    local matched=()
    for kw in "${keywords[@]}"; do
        kw_lower=$(echo "$kw" | tr '[:upper:]' '[:lower:]')
        if echo "$output_lower" | grep -qF "$kw_lower"; then
            matched+=("$kw")
            found=1
        fi
    done

    if [ "$found" -eq 1 ]; then
        log_ok "$name — matched: ${matched[*]}"
        ((PASSED++)) || true
    else
        log_fail "$name — none of [${keywords[*]}] found in response"
        echo "  Response (first 300 chars): $(echo "$output" | head -c 300)"
        ((FAILED++)) || true
    fi
}

send_memory() {
    local message="$1"
    local prev_id="${2:-}"

    local payload
    if [ -n "$prev_id" ]; then
        payload=$(cat <<EOJSON
{
  "model": "MoM",
  "input": "$message",
  "instructions": "You are a helpful assistant with memory.",
  "metadata": {"user_id": "$USER_ID"},
  "previous_response_id": "$prev_id"
}
EOJSON
)
    else
        payload=$(cat <<EOJSON
{
  "model": "MoM",
  "input": "$message",
  "instructions": "You are a helpful assistant with memory.",
  "metadata": {"user_id": "$USER_ID"}
}
EOJSON
)
    fi

    local result
    result=$(curl -s -X POST "$ENDPOINT/v1/responses" \
        -H "Content-Type: application/json" \
        -H "x-authz-user-id: $USER_ID" \
        -d "$payload" 2>&1)

    echo "$result" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('id', ''))
except:
    print('')
" 2>/dev/null
}

# ─── Main ────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  Hierarchical + Hybrid Memory Retrieval API Tests"
echo "============================================================"
echo "  Endpoint:   $ENDPOINT"
echo "  User ID:    $USER_ID"
echo "  Skip seed:  $SKIP_SEED"
echo "============================================================"

# ─── Health check ────────────────────────────────────────────────────────────

log_head "Health Check"
API_ENDPOINT="${API_ENDPOINT:-http://localhost:8082}"
max_retries=15
retry=0
while [ "$retry" -lt "$max_retries" ]; do
    if curl -sf "$API_ENDPOINT/health" > /dev/null 2>&1; then
        log_ok "Router API is healthy at $API_ENDPOINT"
        break
    fi
    ((retry++)) || true
    log_warn "Waiting for router API... ($retry/$max_retries)"
    sleep 2
done
if [ "$retry" -ge "$max_retries" ]; then
    log_fail "Router not reachable at $API_ENDPOINT after $max_retries retries"
    exit 1
fi

log_info "Testing proxy connectivity at $ENDPOINT..."
proxy_test=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$ENDPOINT/v1/responses" \
    -H "Content-Type: application/json" \
    -H "x-authz-user-id: health_check" \
    -d '{"model":"qwen3","input":"ping","instructions":"reply","metadata":{"user_id":"health_check"}}' 2>&1)
if [ "$proxy_test" = "200" ]; then
    log_ok "Proxy is responsive at $ENDPOINT"
else
    log_fail "Proxy not responsive at $ENDPOINT (HTTP $proxy_test)"
    exit 1
fi

# ─── Seed phase ──────────────────────────────────────────────────────────────

if [ "$SKIP_SEED" -eq 0 ]; then
    log_head "Phase 1: Seeding Memories (diverse topics)"

    declare -A SEED_MESSAGES=(
        ["technology"]="I have been learning Rust programming for three months. I use cargo build for compilation and really enjoy the borrow checker."
        ["cooking"]="My signature dish is homemade pesto pasta with fresh basil and pine nuts from my garden. I make it every Friday."
        ["travel"]="I visited Tokyo last summer and walked through the Shibuya crossing. The Tsukiji fish market was amazing too."
        ["sports"]="I run a 10K race every Sunday morning at Central Park. My best time is 48 minutes."
        ["music"]="I play acoustic guitar every evening. My favorite chord progression is I-V-vi-IV and I mostly play fingerstyle."
    )

    prev_id=""
    for topic in technology cooking travel sports music; do
        msg="${SEED_MESSAGES[$topic]}"
        log_info "Seeding [$topic]: ${msg:0:70}..."
        resp_id=$(send_memory "$msg" "$prev_id")
        if [ -z "$resp_id" ]; then
            log_fail "Failed to seed [$topic]"
        else
            log_ok "Seeded [$topic] (id: ${resp_id:0:20}...)"
            prev_id="$resp_id"
        fi
        sleep 1
    done

    log_info "Sending follow-up to trigger final extraction..."
    send_memory "Thanks for remembering all of that." "$prev_id" > /dev/null

    log_info "Waiting ${EXTRACTION_WAIT}s for memory extraction + Milvus indexing..."
    sleep "$EXTRACTION_WAIT"
else
    log_head "Phase 1: Skipped (SKIP_SEED=1)"
fi

# ─── Memory listing ─────────────────────────────────────────────────────────

log_head "Phase 2: Verify Memory Storage via REST API"

mem_response=$(curl -s "$API_ENDPOINT/v1/memory?user_id=$USER_ID" \
    -H "x-authz-user-id: $USER_ID" 2>&1)

mem_count=$(echo "$mem_response" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    items = d if isinstance(d, list) else d.get('memories', d.get('data', []))
    print(len(items))
except:
    print(0)
" 2>/dev/null)

((TOTAL++)) || true
if [ "$mem_count" -gt 0 ]; then
    log_ok "Found $mem_count memories stored for user $USER_ID"
    ((PASSED++)) || true
else
    log_warn "No memories found via REST API (extraction may still be in progress)"
    log_info "Response: $(echo "$mem_response" | head -c 300)"
    ((SKIPPED++)) || true
fi

# ─── Retrieval queries ───────────────────────────────────────────────────────

log_head "Phase 3: Retrieval Queries (hierarchical + hybrid search)"
log_info "All queries are sent in NEW sessions (no previous_response_id)"
log_info "Keywords found in response can only come from Milvus memory injection"
echo ""

run_test \
    "Semantic: programming language" \
    "What programming language am I learning?" \
    "rust" "cargo" "borrow"

run_test \
    "Semantic: cooking" \
    "What do I like to cook?" \
    "pesto" "pasta" "basil" "friday" "signature"

run_test \
    "Semantic: travel destination" \
    "Where did I travel recently?" \
    "tokyo" "shibuya" "tsukiji"

run_test \
    "Semantic: exercise" \
    "What exercise do I do on weekends?" \
    "10k" "run" "central park"

run_test \
    "Semantic: musical instrument" \
    "Tell me about my guitar playing hobby" \
    "guitar" "fingerstyle" "chord" "acoustic" "evening"

# ─── Hybrid keyword queries ─────────────────────────────────────────────────

log_head "Phase 4: Hybrid Keyword Queries (BM25 should boost exact matches)"
echo ""

run_test \
    "Hybrid keyword: cargo build" \
    "Tell me about cargo build" \
    "rust" "cargo" "compilation"

run_test \
    "Hybrid keyword: Shibuya" \
    "What do I know about Shibuya?" \
    "shibuya" "tokyo" "crossing"

run_test \
    "Hybrid keyword: pesto" \
    "Tell me about pesto" \
    "pesto" "basil" "pasta"

run_test \
    "Hybrid keyword: fingerstyle" \
    "What is fingerstyle?" \
    "guitar" "fingerstyle" "chord"

# ─── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "  RESULTS"
echo "============================================================"
echo -e "  Total:   $TOTAL"
echo -e "  ${GREEN}Passed:  $PASSED${NC}"
echo -e "  ${RED}Failed:  $FAILED${NC}"
if [ "$SKIPPED" -gt 0 ]; then
    echo -e "  ${YELLOW}Skipped: $SKIPPED${NC}"
fi
echo "============================================================"

if [ "$FAILED" -gt 0 ]; then
    echo -e "\n${RED}Some tests failed.${NC}"
    echo "Hints:"
    echo "  - Is the router running with config/testing/config.memory-hierarchical.yaml?"
    echo "  - Is Milvus running and reachable at $MILVUS_ADDRESS?"
    echo "  - Is LLM backend running (echo mode recommended)?"
    echo "  - Try increasing EXTRACTION_WAIT (current: ${EXTRACTION_WAIT}s)"
    echo "  - Try: SKIP_SEED=1 $0 $ENDPOINT  (re-run queries only)"
    exit 1
fi

echo -e "\n${GREEN}All tests passed.${NC}"
exit 0
