#!/usr/bin/env bash
set -euo pipefail

# Container runtime (docker or podman) - can be set via environment variable
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# Animation delay
DELAY=0.05

# Function to print colored text
print_color() {
    local color=$1
    local text=$2
    echo -e "${color}${text}${NC}"
}

# Helper functions for common message types
success_msg() {
    print_color "$GREEN" "$1"
}

error_msg() {
    print_color "$RED" "$1"
}

info_msg() {
    print_color "$YELLOW" "$1"
}

section_header() {
    print_color "$CYAN" "$1"
}

# Function to print with typewriter effect
typewriter() {
    local text=$1
    local color=${2:-$WHITE}
    for (( i=0; i<${#text}; i++ )); do
        echo -n -e "${color}${text:$i:1}${NC}"
        sleep $DELAY
    done
    echo
}

# Function to show ASCII art with animation
show_ascii_art() {
    # Skip clear in CI environments (no proper terminal)
    if [ -z "${CI:-}" ]; then
        clear || true
    fi
    echo
    echo
    print_color "$CYAN" "        ██╗   ██╗██╗     ██╗     ███╗   ███╗"
    print_color "$CYAN" "        ██║   ██║██║     ██║     ████╗ ████║"
    print_color "$CYAN" "        ██║   ██║██║     ██║     ██╔████╔██║"
    print_color "$CYAN" "        ╚██╗ ██╔╝██║     ██║     ██║╚██╔╝██║"
    print_color "$CYAN" "         ╚████╔╝ ███████╗███████╗██║ ╚═╝ ██║"
    print_color "$CYAN" "          ╚═══╝  ╚══════╝╚══════╝╚═╝     ╚═╝"
    echo
    print_color "$PURPLE" "      ███████╗███████╗███╗   ███╗ █████╗ ███╗   ██╗████████╗██╗ ██████╗"
    print_color "$PURPLE" "      ██╔════╝██╔════╝████╗ ████║██╔══██╗████╗  ██║╚══██╔══╝██║██╔════╝"
    print_color "$PURPLE" "      ███████╗█████╗  ██╔████╔██║███████║██╔██╗ ██║   ██║   ██║██║     "
    print_color "$PURPLE" "      ╚════██║██╔══╝  ██║╚██╔╝██║██╔══██║██║╚██╗██║   ██║   ██║██║     "
    print_color "$PURPLE" "      ███████║███████╗██║ ╚═╝ ██║██║  ██║██║ ╚████║   ██║   ██║╚██████╗"
    print_color "$PURPLE" "      ╚══════╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝ ╚═════╝"
    echo
    print_color "$YELLOW" "                ██████╗  ██████╗ ██╗   ██╗████████╗███████╗██████╗ "
    print_color "$YELLOW" "                ██╔══██╗██╔═══██╗██║   ██║╚══██╔══╝██╔════╝██╔══██╗"
    print_color "$YELLOW" "                ██████╔╝██║   ██║██║   ██║   ██║   █████╗  ██████╔╝"
    print_color "$YELLOW" "                ██╔══██╗██║   ██║██║   ██║   ██║   ██╔══╝  ██╔══██╗"
    print_color "$YELLOW" "                ██║  ██║╚██████╔╝╚██████╔╝   ██║   ███████╗██║  ██║"
    print_color "$YELLOW" "                ╚═╝  ╚═╝ ╚═════╝  ╚═════╝    ╚═╝   ╚══════╝╚═╝  ╚═╝"
    echo
    echo
    print_color "$GREEN" "                    🚀 Intelligent Request Routing for vLLM 🚀"
    print_color "$WHITE" "                         Quick Start Setup & Launch"
    echo
    sleep 1
}

# Function to show progress bar
show_progress() {
    local current=$1
    local total=$2
    local description=$3
    local width=50
    local percentage=$((current * 100 / total))
    local completed=$((current * width / total))

    printf "\r%s[%s" "${BLUE}" "${GREEN}"
    for ((i=0; i<completed; i++)); do printf "█"; done
    for ((i=completed; i<width; i++)); do printf "░"; done
    printf "%s] %s%% %s%s%s" "${BLUE}" "${percentage}" "${WHITE}" "${description}" "${NC}"

    if [ "$current" -eq "$total" ]; then
        echo
    fi
}

# Function to check prerequisites
check_prerequisites() {
    info_msg "🔍 Checking prerequisites..."
    echo

    local missing_deps=()

    # Check container runtime (docker or podman)
    if ! command -v "$CONTAINER_RUNTIME" &> /dev/null; then
        missing_deps+=("$CONTAINER_RUNTIME")
    fi

    # Check compose command
    if [ "$CONTAINER_RUNTIME" = "podman" ]; then
        if ! command -v podman compose &> /dev/null && ! command -v podman-compose &> /dev/null; then
            missing_deps+=("podman-compose or podman compose plugin")
        fi
    else
        if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
            missing_deps+=("docker-compose")
        fi
    fi

    # Check Make
    if ! command -v make &> /dev/null; then
        missing_deps+=("make")
    fi

    # Check Python (for HuggingFace CLI)
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        missing_deps+=("python3")
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        error_msg "❌ Missing dependencies: ${missing_deps[*]}"
        info_msg "Please install the missing dependencies and try again."
        exit 1
    fi

    success_msg "✅ All prerequisites satisfied!"
    echo
}

# Function to install HuggingFace CLI if needed
install_hf_cli() {
    if ! command -v hf &> /dev/null; then
        info_msg "📦 Installing HuggingFace CLI..."
        pip install huggingface_hub[cli] || pip3 install huggingface_hub[cli]
        success_msg "✅ HuggingFace CLI installed!"
    else
        success_msg "✅ HuggingFace CLI already installed!"
    fi
    echo
}

# Function to download models with progress
download_models() {
    info_msg "📥 Downloading AI models..."
    echo

    # Try full model set first (includes embeddinggemma-300m which requires HF_TOKEN for gated access)
    # If that fails (e.g., 401 on gated models), fall back to minimal set
    if [ "${CI_MINIMAL_MODELS:-}" = "true" ]; then
        info_msg "CI_MINIMAL_MODELS=true detected, using minimal model set"
        export CI_MINIMAL_MODELS=true
    else
        info_msg "Attempting to download full model set (includes embeddinggemma-300m)..."
        if [ -z "${HF_TOKEN:-}" ]; then
            info_msg "ℹ️  Note: HF_TOKEN not set. If embeddinggemma-300m download fails, script will fall back to minimal model set."
            info_msg "   To download Gemma, set HF_TOKEN environment variable: export HF_TOKEN=your_token"
        fi
        export CI_MINIMAL_MODELS=false
    fi

    # Download models and save output to log (visible in real-time)
    if make download-models 2>&1 | tee /tmp/download-models-output.log; then
        success_msg "✅ Models downloaded successfully!"
    else
        # Check if failure was due to gated model (embeddinggemma-300m)
        if grep -q "embeddinggemma.*401\|embeddinggemma.*Unauthorized\|embeddinggemma.*GatedRepoError" /tmp/download-models-output.log 2>/dev/null; then
            info_msg "⚠️  Full model download failed: embeddinggemma-300m requires HF_TOKEN for gated model access"
            info_msg "📋 Falling back to minimal model set (without Gemma)..."
            info_msg "💡 To download Gemma, set HF_TOKEN: export HF_TOKEN=your_token && make download-models"
            export CI_MINIMAL_MODELS=true
            if make download-models 2>&1 | tee /tmp/download-models-output.log; then
                success_msg "✅ Minimal models downloaded successfully!"
                info_msg "ℹ️  Note: Gemma embedding model was skipped. Some features may be limited."
            else
                error_msg "❌ Failed to download even minimal models!"
                info_msg "📋 Check logs: cat /tmp/download-models-output.log"
                exit 1
            fi
        else
            error_msg "❌ Failed to download models!"
            info_msg "📋 Check logs: cat /tmp/download-models-output.log"
            exit 1
        fi
    fi
    echo
}

# Function to start services
start_services() {
    info_msg "🐳 Starting container services (using $CONTAINER_RUNTIME)..."
    echo

    # Start docker-compose services (runs in detached mode via Makefile)
    # Timeout: 600 seconds (10 minutes) to allow for:
    #   - Image pulls (semantic-router, envoy, jaeger, prometheus, grafana, openwebui, pipelines, llm-katan)
    #   - Dashboard build from Dockerfile (Go compilation can take 5-10 minutes)
    #   - Network/system variations
    # Save output to log file for debugging
    if timeout 600 make docker-compose-up CONTAINER_RUNTIME="$CONTAINER_RUNTIME" 2>&1 | tee /tmp/docker-compose-output.log; then
        success_msg "✅ Docker compose command completed!"
        echo "   Output saved to: /tmp/docker-compose-output.log"
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            error_msg "❌ Docker compose command timed out after 10 minutes!"
            info_msg "📋 This might indicate:"
            info_msg "   - Very slow network (image pulls)"
            info_msg "   - System resource constraints"
            info_msg "   - Dashboard build taking too long"
            info_msg "📋 Check logs: cat /tmp/docker-compose-output.log"
        else
            error_msg "❌ Failed to start services!"
            info_msg "📋 Check logs: cat /tmp/docker-compose-output.log"
        fi
        exit 1
    fi
    echo
}

# Function to wait for services to be healthy
wait_for_services() {
    section_header "🔍 Checking service health..."
    local max_attempts=60  
    local attempt=1

    # List of critical services that must be healthy
    local critical_services=("semantic-router" "envoy-proxy")

    while [ $attempt -le $max_attempts ]; do
        local all_healthy=true
        local unhealthy_services=""

        # Check each critical service
        for service in "${critical_services[@]}"; do
            if ! "$CONTAINER_RUNTIME" ps --filter "name=$service" --filter "health=healthy" --format "{{.Names}}" | grep -q "$service" 2>/dev/null; then
                all_healthy=false
                unhealthy_services="$unhealthy_services $service"
            fi
        done

        # Check for any exited/failed containers
        local failed_containers
        failed_containers=$("$CONTAINER_RUNTIME" ps -a --filter "status=exited" --format "{{.Names}}" 2>/dev/null)
        if [ -n "$failed_containers" ]; then
            error_msg "❌ Some containers failed to start: $failed_containers"
            info_msg "📋 Check logs with: $CONTAINER_RUNTIME compose logs $failed_containers"
            return 1
        fi

        if [ "$all_healthy" = true ]; then
            success_msg "✅ All critical services are healthy and ready!"
            echo
            # Show status of all containers
            section_header "📊 Container Status:"
            "$CONTAINER_RUNTIME" ps --format "table {{.Names}}\t{{.Status}}" | grep -E "NAMES|semantic-router|envoy|dashboard|prometheus|grafana|jaeger|openwebui|pipelines|llm-katan"
            echo
            return 0
        fi

        # Show progress every 5 seconds
        if [ $((attempt % 5)) -eq 0 ]; then
            info_msg "⏳ Still waiting for:$unhealthy_services (attempt $attempt/$max_attempts)"
        fi

        sleep 2
        ((attempt++))
    done

    info_msg "⚠️  Timeout: Services are starting but not all are healthy yet."
    print_color "$WHITE" "📋 Check status with: $CONTAINER_RUNTIME ps"
    print_color "$WHITE" "📋 View logs with: $CONTAINER_RUNTIME compose logs -f"
    return 1
}

# Function to show service information
show_service_info() {
    section_header "🌐 Service Information:"
    echo
    print_color "$WHITE" "┌─────────────────────────────────────────────────────────────┐"
    print_color "$WHITE" "│                        🎯 Endpoints                         │"
    print_color "$WHITE" "├─────────────────────────────────────────────────────────────┤"
    print_color "$GREEN" "│  🤖 Semantic Router API:    http://localhost:8801/v1       │"
    print_color "$GREEN" "│  📊 Dashboard:               http://localhost:8700          │"
    print_color "$GREEN" "│  📈 Prometheus:              http://localhost:9090          │"
    print_color "$GREEN" "│  📊 Grafana:                 http://localhost:3000          │"
    print_color "$GREEN" "│  🌐 Open WebUI:              http://localhost:3001          │"
    print_color "$WHITE" "└─────────────────────────────────────────────────────────────┘"
    echo
    section_header "🔧 Useful Commands:"
    echo
    print_color "$WHITE" "  • Check service status:     $CONTAINER_RUNTIME compose ps"
    print_color "$WHITE" "  • View logs:                $CONTAINER_RUNTIME compose logs -f"
    print_color "$WHITE" "  • Stop services:            $CONTAINER_RUNTIME compose down"
    print_color "$WHITE" "  • Restart services:         $CONTAINER_RUNTIME compose restart"
    echo
}

# Function to show completion message
show_completion() {
    echo
    print_color "$CYAN" "╔══════════════════════════════════════════════════════════════════════════════╗"
    print_color "$CYAN" "║                                                                              ║"
    print_color "$GREEN" "║                          🎉 SETUP COMPLETE! 🎉                              ║"
    print_color "$CYAN" "║                                                                              ║"
    print_color "$WHITE" "║  Your vLLM Semantic Router is now running and ready to handle requests!    ║"
    print_color "$CYAN" "║                                                                              ║"
    print_color "$YELLOW" "║  Next steps:                                                                 ║"
    print_color "$WHITE" "║  1. Visit the dashboard: http://localhost:8700                              ║"
    print_color "$WHITE" "║  2. Try the API: http://localhost:8801/v1/models                            ║"
    print_color "$WHITE" "║  3. Monitor with Grafana: http://localhost:3000 (admin/admin)              ║"
    print_color "$CYAN" "║                                                                              ║"
    print_color "$CYAN" "╚══════════════════════════════════════════════════════════════════════════════╝"
    echo

    # Ask if user wants to open browser (skip in CI environments)
    if [ -z "${CI:-}" ]; then
        read -p "$(print_color "$YELLOW" "Would you like to open the dashboard in your browser? (y/N): ")" -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if command -v open &> /dev/null; then
                open http://localhost:8700
            elif command -v xdg-open &> /dev/null; then
                xdg-open http://localhost:8700
            else
                info_msg "Please open http://localhost:8700 in your browser manually."
            fi
        fi
    fi
}

# Main execution
main() {
    # Show ASCII art
    show_ascii_art

    # Check prerequisites
    check_prerequisites

    # Install HuggingFace CLI if needed
    install_hf_cli

    # Download models
    download_models

    # Start services
    start_services

    # Wait for services to be healthy
    if ! wait_for_services; then
        error_msg "❌ Service health check failed or timed out!"
        info_msg "📋 You can check logs with: $CONTAINER_RUNTIME compose logs"
        info_msg "📋 Or continue manually if services are starting"
        exit 1
    fi

    # Show service information
    show_service_info

    # Show completion message
    show_completion
}

# Handle script interruption
trap 'echo; print_color $RED "❌ Setup interrupted!"; exit 1' INT TERM

# Run main function
main "$@"
