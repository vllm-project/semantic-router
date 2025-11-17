#!/bin/bash
# Run comprehensive jailbreak classifier benchmarks

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Jailbreak Classifier Benchmarks ===${NC}"
echo ""

# Check if Rust library is built
if [ ! -f "../candle-binding/target/release/libcandle_semantic_router.a" ]; then
    echo -e "${YELLOW}Building Rust library...${NC}"
    cd ../candle-binding
    cargo build --release
    cd "$SCRIPT_DIR"
fi

# Create results directory
mkdir -p results

# Get timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_FILE="results/jailbreak_bench_${TIMESTAMP}.txt"

echo -e "${GREEN}Running benchmarks...${NC}"
echo -e "Results will be saved to: ${YELLOW}${RESULT_FILE}${NC}"
echo ""

# Run all benchmarks
go test -bench=. -benchmem -benchtime=10s 2>&1 | tee "$RESULT_FILE"

echo ""
echo -e "${GREEN}=== Benchmark Complete ===${NC}"
echo -e "Results saved to: ${YELLOW}${RESULT_FILE}${NC}"
echo ""

# Generate summary
echo -e "${GREEN}=== Quick Summary ===${NC}"
echo ""

# Extract key metrics
echo "Initialization benchmarks:"
grep "BenchmarkInit" "$RESULT_FILE" | awk '{printf "  %-50s %10s ns/op\n", $1, $3}'
echo ""

echo "Classification benchmarks (safe text):"
grep "SafeText-" "$RESULT_FILE" | grep -v "Concurrency" | awk '{printf "  %-50s %10s ns/op\n", $1, $3}'
echo ""

echo "Classification benchmarks (jailbreak text):"
grep "JailbreakText-" "$RESULT_FILE" | grep -v "Concurrency" | awk '{printf "  %-50s %10s ns/op\n", $1, $3}'
echo ""

echo "Concurrent benchmarks (ModernBERT):"
grep "BenchmarkModernBertJailbreak_Concurrent" "$RESULT_FILE" | awk '{printf "  %-50s %10s ns/op\n", $1, $3}'
echo ""

echo "Concurrent benchmarks (DeBERTa):"
grep "BenchmarkDebertaJailbreak_Concurrent" "$RESULT_FILE" | awk '{printf "  %-50s %10s ns/op\n", $1, $3}'
echo ""

echo "Concurrent benchmarks (Unified):"
grep "BenchmarkUnifiedJailbreak_Concurrent" "$RESULT_FILE" | awk '{printf "  %-50s %10s ns/op\n", $1, $3}'
echo ""

# Compare results if previous run exists
mapfile -t files < <(printf '%s\n' results/jailbreak_bench_*.txt 2>/dev/null | sort -r)
PREV_RESULT="${files[1]}"
if [ -n "$PREV_RESULT" ] && [ -f "$PREV_RESULT" ]; then
    echo -e "${GREEN}=== Comparison with Previous Run ===${NC}"
    echo "Previous: $(basename "$PREV_RESULT")"
    echo ""
    
    # Check if benchstat is available
    if command -v benchstat &> /dev/null; then
        benchstat "$PREV_RESULT" "$RESULT_FILE"
    else
        echo -e "${YELLOW}Install benchstat for detailed comparison:${NC}"
        echo "  go install golang.org/x/perf/cmd/benchstat@latest"
    fi
fi

echo ""
echo -e "${GREEN}Done!${NC}"

