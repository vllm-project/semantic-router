#!/bin/bash
# Update performance baselines from benchmark results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERF_DIR="$(dirname "$SCRIPT_DIR")"
BASELINE_DIR="$PERF_DIR/testdata/baselines"
PROJECT_ROOT="$(dirname "$PERF_DIR")"
BENCH_RESULTS="$PROJECT_ROOT/reports/bench-results.txt"

echo "Updating performance baselines..."
echo "Baseline directory: $BASELINE_DIR"
echo "Benchmark results: $BENCH_RESULTS"

# Create baseline directory if it doesn't exist
mkdir -p "$BASELINE_DIR"

# Check if benchmark results exist
if [ ! -f "$BENCH_RESULTS" ]; then
    echo "Error: Benchmark results not found at $BENCH_RESULTS"
    echo "Please run benchmarks first: make perf-baseline-update"
    exit 1
fi

# Get git commit info
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

echo "Parsing benchmark results..."

# Function to parse benchmark metrics
# Format: BenchmarkName-8    iterations    ns/op    B/op    allocs/op
parse_benchmark() {
    local pattern=$1
    local output_file=$2

    {
        echo "{"
        echo "  \"version\": \"v1.0.0\","
        echo "  \"git_commit\": \"$GIT_COMMIT\","
        echo "  \"git_branch\": \"$GIT_BRANCH\","
        echo "  \"timestamp\": \"$TIMESTAMP\","
        echo "  \"benchmarks\": {"
    } > "$output_file"

    local first=true
    while IFS= read -r line; do
        # Match benchmark lines: BenchmarkName-X   iterations   ns/op   ...
        # Note: Uses tabs and may have decimal points in numbers
        if [[ "$line" =~ ^(Benchmark[A-Za-z0-9_/]+)-[0-9]+[[:space:]]+([0-9]+)[[:space:]]+([0-9.]+)[[:space:]]+ns/op[[:space:]]+([0-9]+)[[:space:]]+B/op[[:space:]]+([0-9]+)[[:space:]]+allocs/op ]]; then
            local bench_name="${BASH_REMATCH[1]}"
            local iterations="${BASH_REMATCH[2]}"
            local ns_per_op="${BASH_REMATCH[3]}"
            local bytes_per_op="${BASH_REMATCH[4]}"
            local allocs_per_op="${BASH_REMATCH[5]}"

            # Only include benchmarks matching the pattern
            if [[ "$bench_name" =~ $pattern ]]; then
                if [ "$first" = false ]; then
                    echo "," >> "$output_file"
                fi
                first=false

                {
                    echo -n "    \"$bench_name\": {"
                    echo -n "\"iterations\": $iterations, "
                    echo -n "\"ns_per_op\": $ns_per_op, "
                    echo -n "\"bytes_per_op\": $bytes_per_op, "
                    echo -n "\"allocs_per_op\": $allocs_per_op"
                    echo -n "}"
                } >> "$output_file"
            fi
        fi
    done < "$BENCH_RESULTS"

    {
        echo ""
        echo "  }"
        echo "}"
    } >> "$output_file"
}

echo "Creating baseline files..."

# Classification baseline (BenchmarkClassify*, BenchmarkCGO*)
parse_benchmark "^BenchmarkClassify|^BenchmarkCGO" "$BASELINE_DIR/classification.json"

# Decision baseline (BenchmarkEvaluate*, BenchmarkRule*, BenchmarkPriority*)
parse_benchmark "^BenchmarkEvaluate|^BenchmarkRule|^BenchmarkPriority" "$BASELINE_DIR/decision.json"

# Cache baseline (BenchmarkCache*)
parse_benchmark "^BenchmarkCache" "$BASELINE_DIR/cache.json"

# Extproc baseline (BenchmarkProcess*, BenchmarkHeader*, BenchmarkFull*, BenchmarkDifferent*, BenchmarkConcurrent*)
parse_benchmark "^BenchmarkProcess|^BenchmarkHeader|^BenchmarkFull|^BenchmarkDifferent|^BenchmarkConcurrent" "$BASELINE_DIR/extproc.json"

echo "✓ Baseline files updated successfully"
echo "  Git commit: $GIT_COMMIT"
echo "  Git branch: $GIT_BRANCH"
echo "  Timestamp: $TIMESTAMP"
echo ""
echo "Baseline files created:"
ls -lh "$BASELINE_DIR"/*.json
