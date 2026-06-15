#!/bin/bash
# start-llm-katan.sh - Start LLM Katan servers for testing
#
# This script starts LLM Katan servers using real tiny models
# for testing router classification functionality
#
# Signed-off-by: Yossi Ovadia <yovadia@redhat.com>

set -e

# Configuration
E2E_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS_DIR="$E2E_DIR/logs"
PIDS_FILE="$E2E_DIR/llm_katan_pids.txt"

# Backend type: "transformers" (default), "vllm", or "echo"
# Use "echo" for memory tests - returns the full prompt for verification
BACKEND="${LLM_KATAN_BACKEND:-transformers}"

# Model configurations for LLM Katan servers
# Format: "port:real_model::served_model_name"
LLM_KATAN_MODELS=(
    "8000:Qwen/Qwen3-0.6B::Model-A"
    "8001:Qwen/Qwen3-0.6B::Model-B"
)

# Function to check if LLM Katan is available
check_llm_katan_available() {
    if command -v llm-katan >/dev/null 2>&1; then
        return 0
    elif python -c "import llm_katan" >/dev/null 2>&1; then
        return 0
    else
        echo "❌ Error: LLM Katan not found. Please install with:"
        echo "   pip install llm-katan"
        exit 1
    fi
}

# Function to check if port is already in use
check_port() {
    local port=$1
    if lsof -Pi :"$port" -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "Port $port is already in use"
        return 1
    fi
    return 0
}

# Function to start servers in foreground for development
start_servers_foreground() {
    echo "Starting LLM Katan servers in FOREGROUND mode..."
    echo "==============================================="
    echo "Press Ctrl+C to stop all servers"
    echo "==============================================="

    # Check prerequisites
    check_llm_katan_available

    # Create logs directory
    mkdir -p "$LOGS_DIR"

    # Check if ports are available
    for model_config in "${LLM_KATAN_MODELS[@]}"; do
        port="${model_config%%:*}"
        if ! check_port "$port"; then
            echo "Error: Port $port is already in use. Please stop existing services."
            exit 1
        fi
    done

    # Array to store background process PIDs
    declare -a PIDS=()

    # Start servers in background but show output
    for model_config in "${LLM_KATAN_MODELS[@]}"; do
        port="${model_config%%:*}"
        model_spec="${model_config#*:}"
        real_model="${model_spec%%::*}"
        served_name="${model_spec##*::}"

        echo "🚀 Starting LLM Katan server on port $port..."
        echo "   Backend: $BACKEND"
        echo "   Real model: $real_model"
        echo "   Served as: $served_name"

        # Start server and capture PID
        # For echo backend, model is ignored but still required by CLI
        llm-katan \
            --model "$real_model" \
            --served-model-name "$served_name" \
            --port "$port" \
            --host 127.0.0.1 \
            --max-tokens 512 \
            --temperature 0.7 \
            --backend "$BACKEND" \
            --log-level DEBUG &
        local pid=$!
        PIDS+=("$pid")
        echo "$pid" >> "$PIDS_FILE"

        echo "   ✅ Server started on port $port (PID: $pid)"
    done

    echo ""
    echo "🤖 LLM Katan servers are running!"
    echo "Server endpoints:"
    for model_config in "${LLM_KATAN_MODELS[@]}"; do
        port="${model_config%%:*}"
        model_spec="${model_config#*:}"
        served_name="${model_spec##*::}"
        echo "  📡 http://127.0.0.1:$port (served as: $served_name)"
    done
    echo ""
    echo "🔍 You'll see request logs below as they come in..."
    echo "🛑 Press Ctrl+C to stop all servers"
    printf '=%.0s' {1..50}
    echo ""
    echo ""

    # Function to cleanup on exit
    cleanup() {
        echo ""
        echo "🛑 Stopping all LLM Katan servers..."
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "   Stopping PID $pid..."
                kill "$pid" 2>/dev/null || true
            fi
        done
        # Clean up PID file
        rm -f "$PIDS_FILE"
        echo "✅ All LLM Katan servers stopped"
        exit 0
    }

    # Set up signal handlers
    trap cleanup SIGINT SIGTERM

    # Wait for all background processes
    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done
}

# Main execution - always run in foreground mode
start_servers_foreground
