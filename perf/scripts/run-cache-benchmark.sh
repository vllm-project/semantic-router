#!/bin/bash

# Semantic Cache Benchmark Runner Script
# Provides an easy interface to run various benchmark scenarios

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCHMARK_DIR="$PROJECT_ROOT/src/semantic-router/cmd/cache-benchmark"
BENCHMARK_BIN="$BENCHMARK_DIR/cache-benchmark"

# Print colored message
print_info() {
    echo -e "${BLUE}ℹ ${NC}$1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Print banner
print_banner() {
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════════════════════════════╗"
    echo "║                      SEMANTIC CACHE PERFORMANCE BENCHMARK                             ║"
    echo "║                                                                                       ║"
    echo "║  Evaluates latency and throughput for semantic cache search operations               ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════════════════╝"
    echo ""
}

# Build the benchmark tool
build_benchmark() {
    print_info "Building benchmark tool..."
    cd "$BENCHMARK_DIR"
    
    if ! CGO_ENABLED=1 go build -o cache-benchmark -tags cgo; then
        print_error "Failed to build benchmark tool"
        exit 1
    fi
    
    print_success "Benchmark tool built successfully"
}

# Check if benchmark binary exists, build if not
ensure_binary() {
    if [ ! -f "$BENCHMARK_BIN" ]; then
        print_warning "Benchmark binary not found, building..."
        build_benchmark
    fi
}

# Show usage
show_usage() {
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  quick                - Run quick benchmark (default)"
    echo "  full                 - Run full benchmark suite"
    echo "  scalability          - Run scalability analysis"
    echo "  concurrency          - Run concurrency impact analysis"
    echo "  component            - Show component latency breakdown"
    echo "  stress               - Stress test with high concurrency"
    echo "  compare-hnsw         - Compare HNSW vs Linear search"
    echo "  profile              - Run with CPU and memory profiling"
    echo "  custom               - Run with custom parameters"
    echo "  build                - Build the benchmark tool"
    echo "  clean                - Clean benchmark artifacts"
    echo "  help                 - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 quick             # Run quick benchmark"
    echo "  $0 full              # Run full benchmark suite"
    echo "  $0 scalability       # Analyze cache scalability"
    echo "  $0 custom --cache-sizes=1000 --concurrency=50 --requests=2000"
    echo ""
}

# Run quick benchmark
run_quick() {
    print_info "Running quick benchmark..."
    ensure_binary
    cd "$BENCHMARK_DIR"
    ./cache-benchmark --quick "$@"
}

# Run full benchmark
run_full() {
    print_info "Running full benchmark suite..."
    print_warning "This may take 15-30 minutes depending on your hardware"
    ensure_binary
    cd "$BENCHMARK_DIR"
    ./cache-benchmark --full "$@"
}

# Run scalability analysis
run_scalability() {
    print_info "Running scalability analysis..."
    ensure_binary
    cd "$BENCHMARK_DIR"
    ./cache-benchmark --scalability "$@"
}

# Run concurrency analysis
run_concurrency() {
    print_info "Running concurrency impact analysis..."
    ensure_binary
    cd "$BENCHMARK_DIR"
    ./cache-benchmark --concurrency-test "$@"
}

# Run component breakdown
run_component() {
    print_info "Running component latency breakdown..."
    ensure_binary
    cd "$BENCHMARK_DIR"
    ./cache-benchmark --component-breakdown "$@"
}

# Run stress test
run_stress() {
    print_info "Running stress test with high concurrency..."
    print_warning "This will push your system hard!"
    ensure_binary
    cd "$BENCHMARK_DIR"
    ./cache-benchmark \
        --cache-sizes=10000 \
        --concurrency=100,200,500 \
        --requests=5000 \
        --concurrency-test \
        "$@"
}

# Compare HNSW vs Linear
compare_hnsw() {
    print_info "Comparing HNSW vs Linear search..."
    ensure_binary
    cd "$BENCHMARK_DIR"
    
    print_info "Running with HNSW enabled..."
    ./cache-benchmark \
        --cache-sizes=1000,10000 \
        --concurrency=50,100 \
        --requests=2000 \
        --hnsw=true \
        --json \
        --output=hnsw_results.json \
        "$@"
    
    print_info "Running with Linear search..."
    ./cache-benchmark \
        --cache-sizes=1000,10000 \
        --concurrency=50,100 \
        --requests=2000 \
        --hnsw=false \
        --json \
        --output=linear_results.json \
        "$@"
    
    print_success "Results saved to:"
    echo "  - HNSW: $BENCHMARK_DIR/hnsw_results.json"
    echo "  - Linear: $BENCHMARK_DIR/linear_results.json"
}

# Run with profiling
run_profile() {
    print_info "Running benchmark with profiling enabled..."
    ensure_binary
    cd "$BENCHMARK_DIR"
    ./cache-benchmark \
        --quick \
        --cpuprofile=cpu.prof \
        --memprofile=mem.prof \
        "$@"
    
    print_success "Profiles saved to:"
    echo "  - CPU: $BENCHMARK_DIR/cpu.prof"
    echo "  - Memory: $BENCHMARK_DIR/mem.prof"
    echo ""
    echo "To analyze profiles, run:"
    echo "  go tool pprof -http=:8080 $BENCHMARK_DIR/cpu.prof"
    echo "  go tool pprof -http=:8081 $BENCHMARK_DIR/mem.prof"
}

# Run custom benchmark
run_custom() {
    print_info "Running custom benchmark..."
    ensure_binary
    cd "$BENCHMARK_DIR"
    ./cache-benchmark "$@"
}

# Clean artifacts
clean() {
    print_info "Cleaning benchmark artifacts..."
    cd "$BENCHMARK_DIR"
    rm -f cache-benchmark ./*.prof ./*.json
    print_success "Cleaned benchmark artifacts"
}

# Main script logic
main() {
    print_banner
    
    # Check if Go is installed
    if ! command -v go &> /dev/null; then
        print_error "Go is not installed. Please install Go 1.19 or later."
        exit 1
    fi
    
    # Check if CGO is available
    if [ "$CGO_ENABLED" = "0" ]; then
        print_warning "CGO_ENABLED is set to 0. Setting it to 1 for benchmark builds."
        export CGO_ENABLED=1
    fi
    
    # Set model paths if not already set
    if [ -z "$QWEN3_MODEL_PATH" ]; then
        export QWEN3_MODEL_PATH="$PROJECT_ROOT/models/mom-embedding-pro"
        print_info "QWEN3_MODEL_PATH set to: $QWEN3_MODEL_PATH"
    fi
    
    # Show current configuration
    echo ""
    print_info "Current Configuration:"
    echo "  USE_GPU:           ${USE_GPU:-false}"
    echo "  USE_HNSW:          ${USE_HNSW:-(from config)}"
    echo "  QWEN3_MODEL_PATH:  $QWEN3_MODEL_PATH"
    echo ""
    print_info "To override: export USE_GPU=true USE_HNSW=true"
    echo ""
    
    # Set library path for candle binding
    if [ -z "$LD_LIBRARY_PATH" ]; then
        export LD_LIBRARY_PATH="$PROJECT_ROOT/candle-binding/target/release"
    else
        export LD_LIBRARY_PATH="$PROJECT_ROOT/candle-binding/target/release:$LD_LIBRARY_PATH"
    fi
    
    # Parse command
    COMMAND="${1:-quick}"
    shift || true
    
    case "$COMMAND" in
        quick)
            run_quick "$@"
            ;;
        full)
            run_full "$@"
            ;;
        scalability)
            run_scalability "$@"
            ;;
        concurrency)
            run_concurrency "$@"
            ;;
        component)
            run_component "$@"
            ;;
        stress)
            run_stress "$@"
            ;;
        compare-hnsw)
            compare_hnsw "$@"
            ;;
        profile)
            run_profile "$@"
            ;;
        custom)
            run_custom "$@"
            ;;
        build)
            build_benchmark
            ;;
        clean)
            clean
            ;;
        help|--help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            echo ""
            show_usage
            exit 1
            ;;
    esac
    
    print_success "Benchmark completed!"
}

# Run main function
main "$@"

