#!/bin/bash
# Hallucination Detection Demo - End-to-End Test
# This script starts all services and runs the interactive demo

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Ports
MOCK_VLLM_PORT=8002
MOCK_SEARCH_PORT=8003
ROUTER_PORT=8801
WEB_CLIENT_PORT=8888

# Check for --web flag
USE_WEB=false
for arg in "$@"; do
    if [ "$arg" = "--web" ]; then
        USE_WEB=true
    fi
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}=============================================="
echo "  Hallucination Detection E2E Demo"
echo -e "==============================================${NC}"
echo ""

cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    if [ -f /tmp/mock_vllm_demo.pid ]; then kill "$(cat /tmp/mock_vllm_demo.pid)" 2>/dev/null || true; fi
    if [ -f /tmp/mock_search_demo.pid ]; then kill "$(cat /tmp/mock_search_demo.pid)" 2>/dev/null || true; fi
    if [ -f /tmp/router_demo.pid ]; then kill "$(cat /tmp/router_demo.pid)" 2>/dev/null || true; fi
    if [ -f /tmp/envoy_demo.pid ]; then kill "$(cat /tmp/envoy_demo.pid)" 2>/dev/null || true; fi
    rm -f /tmp/mock_vllm_demo.pid /tmp/mock_search_demo.pid /tmp/router_demo.pid /tmp/envoy_demo.pid
    # Kill any remaining processes on our ports
    lsof -ti:$MOCK_VLLM_PORT | xargs kill -9 2>/dev/null || true
    lsof -ti:$MOCK_SEARCH_PORT | xargs kill -9 2>/dev/null || true
    lsof -ti:$ROUTER_PORT | xargs kill -9 2>/dev/null || true
    lsof -ti:50051 | xargs kill -9 2>/dev/null || true
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true
    pkill -f "func-e" 2>/dev/null || true
    echo -e "${GREEN}Done.${NC}"
}
trap cleanup EXIT

# Pre-cleanup: kill any processes using our ports
echo -e "${YELLOW}[0/4]${NC} Cleaning up any existing processes..."
lsof -ti:$MOCK_VLLM_PORT | xargs kill -9 2>/dev/null || true
lsof -ti:$MOCK_SEARCH_PORT | xargs kill -9 2>/dev/null || true
lsof -ti:$ROUTER_PORT | xargs kill -9 2>/dev/null || true
lsof -ti:50051 | xargs kill -9 2>/dev/null || true
lsof -ti:8080 | xargs kill -9 2>/dev/null || true
sleep 1
echo -e "   ${GREEN}✓ Cleanup complete${NC}"

# Step 1: Start Mock vLLM with tool calling
echo -e "${YELLOW}[1/4]${NC} Starting Mock vLLM server (port $MOCK_VLLM_PORT)..."
python3 "$SCRIPT_DIR/mock_vllm_toolcall.py" --port $MOCK_VLLM_PORT > /tmp/mock_vllm_demo.log 2>&1 &
echo $! > /tmp/mock_vllm_demo.pid
sleep 1
if curl -sf http://127.0.0.1:$MOCK_VLLM_PORT/health > /dev/null; then
    echo -e "   ${GREEN}✓ Mock vLLM is healthy${NC}"
else
    echo -e "   ${RED}✗ Mock vLLM failed to start${NC}"
    cat /tmp/mock_vllm_demo.log
    exit 1
fi

# Step 2: Start Mock Web Search
echo -e "${YELLOW}[2/4]${NC} Starting Mock Web Search server (port $MOCK_SEARCH_PORT)..."
python3 "$SCRIPT_DIR/mock_web_search.py" --port $MOCK_SEARCH_PORT > /tmp/mock_search_demo.log 2>&1 &
echo $! > /tmp/mock_search_demo.pid
sleep 1
if curl -sf http://127.0.0.1:$MOCK_SEARCH_PORT/health > /dev/null; then
    echo -e "   ${GREEN}✓ Mock Web Search is healthy${NC}"
else
    echo -e "   ${RED}✗ Mock Web Search failed to start${NC}"
    cat /tmp/mock_search_demo.log
    exit 1
fi

# Step 3: Start Semantic Router (gRPC ExtProc on port 50051)
echo -e "${YELLOW}[3/5]${NC} Starting Semantic Router (ExtProc port 50051)..."
cd "$ROOT_DIR"
export LD_LIBRARY_PATH=${ROOT_DIR}/candle-binding/target/release
nohup ./bin/router -config=config/testing/config.hallucination.yaml > /tmp/router_demo.log 2>&1 &
echo $! > /tmp/router_demo.pid

echo "   Waiting for router to initialize models (15s)..."
sleep 15

# Check if router initialized
if grep -q "Fact-check classifier initialized" /tmp/router_demo.log 2>/dev/null; then
    echo -e "   ${GREEN}✓ Router models initialized${NC}"
else
    echo -e "   ${YELLOW}⚠ Router may still be initializing...${NC}"
    echo "   Check /tmp/router_demo.log for details"
fi

# Step 4: Start Envoy proxy (HTTP to gRPC)
echo -e "${YELLOW}[4/5]${NC} Starting Envoy proxy (port $ROUTER_PORT)..."
if ! command -v func-e >/dev/null 2>&1; then
    echo "   Installing func-e..."
    curl -sL https://func-e.io/install.sh | bash -s -- -b /usr/local/bin
fi
nohup func-e run --config-path config/envoy.yaml > /tmp/envoy_demo.log 2>&1 &
echo $! > /tmp/envoy_demo.pid
sleep 3
if curl -sf http://127.0.0.1:$ROUTER_PORT/v1/models > /dev/null 2>&1; then
    echo -e "   ${GREEN}✓ Envoy is ready${NC}"
else
    echo -e "   ${GREEN}✓ Envoy started${NC}"
fi

# Step 5: Run Chat Client
echo ""
echo -e "${CYAN}=============================================="
echo "  All services started!"
echo -e "==============================================${NC}"
echo ""
echo "Services running:"
echo "  • Mock vLLM:      http://127.0.0.1:$MOCK_VLLM_PORT"
echo "  • Mock Search:    http://127.0.0.1:$MOCK_SEARCH_PORT"
echo "  • Router gRPC:    localhost:50051"
echo "  • Envoy HTTP:     http://127.0.0.1:$ROUTER_PORT"
echo ""

if [ "$USE_WEB" = true ]; then
    echo -e "${YELLOW}[5/5]${NC} Starting web-based chat client..."
    echo ""
    echo -e "${GREEN}🌐 Open in browser: http://localhost:$WEB_CLIENT_PORT${NC}"
    echo ""
    python3 "$SCRIPT_DIR/web_client.py" \
        --port "$WEB_CLIENT_PORT" \
        --router-url "http://127.0.0.1:$ROUTER_PORT" \
        --search-url "http://127.0.0.1:$MOCK_SEARCH_PORT"
else
    echo -e "${YELLOW}[5/5]${NC} Starting CLI chat client..."
    echo -e "   ${CYAN}(Use --web flag for browser-based UI)${NC}"
    echo ""
    # Filter out our custom flags before passing to chat_client
    FILTERED_ARGS=()
    for arg in "$@"; do
        if [ "$arg" != "--web" ]; then
            FILTERED_ARGS+=("$arg")
        fi
    done
    python3 "$SCRIPT_DIR/chat_client.py" \
        --router-url "http://127.0.0.1:$ROUTER_PORT" \
        --search-url "http://127.0.0.1:$MOCK_SEARCH_PORT" \
        "${FILTERED_ARGS[@]}"
fi
