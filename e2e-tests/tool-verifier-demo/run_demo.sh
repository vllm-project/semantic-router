#!/bin/bash
# Tool Verification E2E Demo
# This script starts all services and runs the interactive demo
#
# Usage:
#   ./run_demo.sh          # Interactive mode
#   ./run_demo.sh --demo   # Run predefined scenarios
#   ./run_demo.sh --test   # Run automated tests with assertions

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Ports
MOCK_VLLM_PORT=8002
ROUTER_PORT=8801
ENVOY_ADMIN_PORT=9901

# Parse arguments
RUN_MODE=""
for arg in "$@"; do
    case "$arg" in
        --demo) RUN_MODE="--demo" ;;
        --test) RUN_MODE="--test" ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}=============================================="
echo "  Tool Verification E2E Demo"
echo "  Two-Stage Jailbreak Detection Pipeline"
echo -e "==============================================${NC}"
echo ""

cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    [ -f /tmp/mock_vllm_tv.pid ] && kill "$(cat /tmp/mock_vllm_tv.pid)" 2>/dev/null || true
    [ -f /tmp/router_tv.pid ] && kill "$(cat /tmp/router_tv.pid)" 2>/dev/null || true
    [ -f /tmp/envoy_tv.pid ] && kill "$(cat /tmp/envoy_tv.pid)" 2>/dev/null || true
    rm -f /tmp/mock_vllm_tv.pid /tmp/router_tv.pid /tmp/envoy_tv.pid
    # Kill any remaining processes on our ports
    lsof -ti:$MOCK_VLLM_PORT | xargs kill -9 2>/dev/null || true
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
lsof -ti:$ROUTER_PORT | xargs kill -9 2>/dev/null || true
lsof -ti:50051 | xargs kill -9 2>/dev/null || true
lsof -ti:8080 | xargs kill -9 2>/dev/null || true
sleep 1
echo -e "   ${GREEN}✓ Cleanup complete${NC}"

# Step 1: Check if models are available
echo -e "${YELLOW}[1/4]${NC} Checking model availability..."
STAGE1_MODEL="$ROOT_DIR/models/function-call-sentinel"
STAGE2_MODEL="$ROOT_DIR/models/tool-call-verifier"

if [ ! -d "$STAGE1_MODEL" ]; then
    echo -e "   ${YELLOW}⚠ Stage 1 model not found at $STAGE1_MODEL${NC}"
    echo -e "   ${CYAN}Download with: huggingface-cli download rootfs/function-call-sentinel --local-dir $STAGE1_MODEL${NC}"
    echo -e "   ${YELLOW}Continuing anyway (may fail at runtime)...${NC}"
else
    echo -e "   ${GREEN}✓ Stage 1 model found${NC}"
fi

if [ ! -d "$STAGE2_MODEL" ]; then
    echo -e "   ${YELLOW}⚠ Stage 2 model not found at $STAGE2_MODEL${NC}"
    echo -e "   ${CYAN}Download with: huggingface-cli download rootfs/tool-call-verifier --local-dir $STAGE2_MODEL${NC}"
    echo -e "   ${YELLOW}Continuing anyway (may fail at runtime)...${NC}"
else
    echo -e "   ${GREEN}✓ Stage 2 model found${NC}"
fi

# Step 2: Start Mock vLLM with tool calling
echo -e "${YELLOW}[2/4]${NC} Starting Mock vLLM server (port $MOCK_VLLM_PORT)..."
python3 "$SCRIPT_DIR/mock_vllm_toolcall.py" --port $MOCK_VLLM_PORT > /tmp/mock_vllm_tv.log 2>&1 &
echo $! > /tmp/mock_vllm_tv.pid
sleep 1
if curl -sf http://127.0.0.1:$MOCK_VLLM_PORT/health > /dev/null; then
    echo -e "   ${GREEN}✓ Mock vLLM is healthy${NC}"
else
    echo -e "   ${RED}✗ Mock vLLM failed to start${NC}"
    cat /tmp/mock_vllm_tv.log
    exit 1
fi

# Step 3: Start Semantic Router (gRPC ExtProc on port 50051)
echo -e "${YELLOW}[3/4]${NC} Starting Semantic Router (ExtProc port 50051)..."
cd "$ROOT_DIR"

# Check if binary exists
if [ ! -f "./bin/router" ]; then
    echo -e "   ${RED}✗ Router binary not found at ./bin/router${NC}"
    echo -e "   ${CYAN}Build with: cd src/semantic-router && go build -o ../../bin/router${NC}"
    exit 1
fi

export LD_LIBRARY_PATH=${ROOT_DIR}/candle-binding/target/release
nohup ./bin/router -config=config/testing/config.toolverifier.yaml > /tmp/router_tv.log 2>&1 &
echo $! > /tmp/router_tv.pid

echo "   Waiting for router to initialize models (15s)..."
sleep 15

# Check if router initialized Stage 1
if grep -q "Tool Verifier Stage 1 initialized" /tmp/router_tv.log 2>/dev/null; then
    echo -e "   ${GREEN}✓ Stage 1 (FunctionCallSentinel) initialized${NC}"
else
    echo -e "   ${YELLOW}⚠ Stage 1 may still be initializing...${NC}"
fi

# Check if router initialized Stage 2
if grep -q "Tool Verifier Stage 2 initialized" /tmp/router_tv.log 2>/dev/null; then
    echo -e "   ${GREEN}✓ Stage 2 (ToolCallVerifier) initialized${NC}"
else
    echo -e "   ${YELLOW}⚠ Stage 2 may still be initializing...${NC}"
fi

# Check for errors
if grep -q "ERROR\|FATAL\|panic" /tmp/router_tv.log 2>/dev/null; then
    echo -e "   ${RED}⚠ Errors detected in router log:${NC}"
    grep "ERROR\|FATAL\|panic" /tmp/router_tv.log | head -5
fi

echo "   Check /tmp/router_tv.log for details"

# Step 4: Start Envoy proxy (HTTP to gRPC)
echo -e "${YELLOW}[4/4]${NC} Starting Envoy proxy (port $ROUTER_PORT)..."
if ! command -v func-e >/dev/null 2>&1; then
    echo "   Installing func-e..."
    curl -sL https://func-e.io/install.sh | bash -s -- -b /usr/local/bin
fi
nohup func-e run --config-path config/envoy.yaml > /tmp/envoy_tv.log 2>&1 &
echo $! > /tmp/envoy_tv.pid
sleep 3
if curl -sf http://127.0.0.1:$ROUTER_PORT/v1/models > /dev/null 2>&1; then
    echo -e "   ${GREEN}✓ Envoy is ready${NC}"
else
    echo -e "   ${GREEN}✓ Envoy started${NC}"
fi

# All services ready
echo ""
echo -e "${CYAN}=============================================="
echo "  All services started!"
echo -e "==============================================${NC}"
echo ""
echo "Services running:"
echo "  • Mock vLLM:      http://127.0.0.1:$MOCK_VLLM_PORT"
echo "  • Router gRPC:    localhost:50051"
echo "  • Envoy HTTP:     http://127.0.0.1:$ROUTER_PORT"
echo ""
echo "Tool Verification Pipeline:"
echo "  • Stage 1: FunctionCallSentinel (prompt classification)"
echo "  • Stage 2: ToolCallVerifier (tool-call authorization)"
echo ""
echo "Logs:"
echo "  • Mock vLLM:  /tmp/mock_vllm_tv.log"
echo "  • Router:     /tmp/router_tv.log"
echo "  • Envoy:      /tmp/envoy_tv.log"
echo ""

# Run the client
if [ "$RUN_MODE" = "--test" ]; then
    echo -e "${YELLOW}Running automated tests...${NC}"
    echo ""
    python3 "$SCRIPT_DIR/chat_client.py" --router-url "http://127.0.0.1:$ROUTER_PORT" --test
    TEST_RESULT=$?
    echo ""
    if [ $TEST_RESULT -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
    else
        echo -e "${RED}Some tests failed!${NC}"
    fi
    exit $TEST_RESULT
elif [ "$RUN_MODE" = "--demo" ]; then
    echo -e "${YELLOW}Running demo scenarios...${NC}"
    python3 "$SCRIPT_DIR/chat_client.py" --router-url "http://127.0.0.1:$ROUTER_PORT" --demo
else
    echo -e "${YELLOW}Starting interactive client...${NC}"
    echo ""
    python3 "$SCRIPT_DIR/chat_client.py" --router-url "http://127.0.0.1:$ROUTER_PORT"
fi

