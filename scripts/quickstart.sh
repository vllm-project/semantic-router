#!/usr/bin/env bash
set -euo pipefail

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
    clear
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
    print_color "$YELLOW" "🔍 Checking prerequisites..."
    echo

    local missing_deps=()

    # Check Docker
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi

    # Check Docker Compose
    if ! command -v docker compose &> /dev/null && ! command -v docker-compose &> /dev/null; then
        missing_deps+=("docker-compose")
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
        print_color "$RED" "❌ Missing dependencies: ${missing_deps[*]}"
        print_color "$YELLOW" "Please install the missing dependencies and try again."
        exit 1
    fi

    print_color "$GREEN" "✅ All prerequisites satisfied!"
    echo
}

# Function to install HuggingFace CLI if needed
install_hf_cli() {
    if ! command -v hf &> /dev/null; then
        print_color "$YELLOW" "📦 Installing HuggingFace CLI..."
        pip install huggingface_hub[cli] || pip3 install huggingface_hub[cli]
        print_color "$GREEN" "✅ HuggingFace CLI installed!"
    else
        print_color "$GREEN" "✅ HuggingFace CLI already installed!"
    fi
    echo
}

# Function to download models with progress
download_models() {
    print_color "$YELLOW" "📥 Downloading AI models..."
    echo

    # Use minimal model set for faster setup
    export CI_MINIMAL_MODELS=false

    # Start the download process with filtered output
    make download-models 2>&1 | grep -E "(downloading|downloaded|Downloaded|✓|✅|❌|Error|error|Failed|failed|CI_MINIMAL_MODELS|Running download-models)" | while IFS= read -r line; do
        # Filter out verbose HuggingFace download progress
        if [[ ! "$line" =~ (Fetching|\.safetensors|\.json|\.txt|\.bin|B/s|%|/s) ]]; then
            # Suppress output - no information displayed
            :
        fi
    done

    if make download-models > /dev/null 2>&1; then
        print_color "$GREEN" "✅ Models downloaded successfully!"
    else
        print_color "$RED" "❌ Failed to download models!"
        exit 1
    fi
    echo
}

# Function to start services
start_services() {
    print_color "$YELLOW" "🐳 Starting Docker services..."
    echo

    # Start docker-compose services with filtered output
    make docker-compose-up 2>&1 | grep -E "(Running docker-compose-up|Starting services|Container.*Running|Container.*Healthy|Container.*Started|✓|✅|❌|Error|error|Failed|failed)" | while IFS= read -r line; do
        # Show only key status updates
        if [[ "$line" =~ (Container.*Running|Container.*Healthy|Starting services|Running docker-compose-up) ]]; then
            echo "   $line"
        fi
    done

    if make docker-compose-up > /dev/null 2>&1; then
        print_color "$GREEN" "✅ Services started successfully!"
    else
        print_color "$RED" "❌ Failed to start services!"
        exit 1
    fi
    echo
}

# Function to wait for services to be healthy
wait_for_services() {
    # Silently wait for services to become healthy
    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        # Check if semantic-router container is healthy
        if docker ps --filter "name=semantic-router" --filter "health=healthy" --format "{{.Names}}" | grep -q "semantic-router" 2>/dev/null; then
            print_color "$GREEN" "✅ All services are healthy and ready!"
            return 0
        fi

        sleep 2
        ((attempt++))
    done

    print_color "$YELLOW" "⚠️  Services are starting but may not be fully healthy yet."
    print_color "$WHITE" "You can check the status with: docker compose ps"
}

# Function to show service information
show_service_info() {
    print_color "$CYAN" "🌐 Service Information:"
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
    print_color "$CYAN" "🔧 Useful Commands:"
    echo
    print_color "$WHITE" "  • Check service status:     docker compose ps"
    print_color "$WHITE" "  • View logs:                docker compose logs -f"
    print_color "$WHITE" "  • Stop services:            docker compose down"
    print_color "$WHITE" "  • Restart services:         docker compose restart"
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

    # Ask if user wants to open browser
    read -p "$(print_color "$YELLOW" "Would you like to open the dashboard in your browser? (y/N): ")" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if command -v open &> /dev/null; then
            open http://localhost:8700
        elif command -v xdg-open &> /dev/null; then
            xdg-open http://localhost:8700
        else
            print_color "$YELLOW" "Please open http://localhost:8700 in your browser manually."
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
    wait_for_services

    # Show service information
    show_service_info

    # Show completion message
    show_completion
}

# Handle script interruption
trap 'echo; print_color $RED "❌ Setup interrupted!"; exit 1' INT TERM

# Run main function
main "$@"
