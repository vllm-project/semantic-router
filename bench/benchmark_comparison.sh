#!/bin/bash

# Multi-Dataset Reasoning Benchmark Comparison
# 
# Comprehensive evaluation framework comparing semantic router performance
# against direct vLLM inference across reasoning datasets.
#
# Usage: ./benchmark_comparison.sh [dataset] [samples_per_category] [concurrent_requests]
# Example: ./benchmark_comparison.sh gpqa 5 2

set -e

# Configuration parameters
DATASET=${1:-"arc"}
SAMPLES_PER_CATEGORY=${2:-5}
CONCURRENT_REQUESTS=${3:-2}

# Semantic router configuration
ROUTER_ENDPOINT="http://127.0.0.1:8801/v1"
ROUTER_API_KEY="1234"
ROUTER_MODEL="auto"

# Direct vLLM configuration
VLLM_ENDPOINT="http://127.0.0.1:8000/v1"
VLLM_API_KEY="1234"
VLLM_MODEL="openai/gpt-oss-20b"

# Evaluation parameters
TEMPERATURE=0.0
OUTPUT_DIR="results/comparison_$(date +%Y%m%d_%H%M%S)"

echo "🎯 MULTI-DATASET REASONING BENCHMARK"
echo "====================================="
echo "Dataset: $DATASET"
echo "Samples per category: $SAMPLES_PER_CATEGORY"
echo "Concurrent requests: $CONCURRENT_REQUESTS"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Ensure we're in the bench directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -f "../.venv/bin/activate" ]; then
    echo "📦 Activating virtual environment..."
    source ../.venv/bin/activate
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "🔄 PHASE 1: ROUTER EVALUATION (via Envoy)"
echo "------------------------------------------"
echo "Endpoint: $ROUTER_ENDPOINT"
echo "Model: $ROUTER_MODEL (router decides)"
echo ""

# Run router benchmark
python3 -m vllm_semantic_router_bench.router_reason_bench_multi_dataset \
    --dataset "$DATASET" \
    --samples-per-category "$SAMPLES_PER_CATEGORY" \
    --concurrent-requests "$CONCURRENT_REQUESTS" \
    --router-endpoint "$ROUTER_ENDPOINT" \
    --router-api-key "$ROUTER_API_KEY" \
    --router-models "$ROUTER_MODEL" \
    --temperature "$TEMPERATURE" \
    --output-dir "$OUTPUT_DIR" \
    --run-router

echo ""
echo "🔄 PHASE 2: DIRECT vLLM EVALUATION"
echo "-----------------------------------"
echo "Endpoint: $VLLM_ENDPOINT"
echo "Model: $VLLM_MODEL (direct access)"
echo ""

# Run direct vLLM benchmark
python3 -m vllm_semantic_router_bench.router_reason_bench_multi_dataset \
    --dataset "$DATASET" \
    --samples-per-category "$SAMPLES_PER_CATEGORY" \
    --concurrent-requests "$CONCURRENT_REQUESTS" \
    --vllm-endpoint "$VLLM_ENDPOINT" \
    --vllm-api-key "$VLLM_API_KEY" \
    --vllm-models "$VLLM_MODEL" \
    --vllm-exec-modes "NR" "XC" \
    --temperature "$TEMPERATURE" \
    --output-dir "$OUTPUT_DIR" \
    --run-vllm

echo ""
echo "🎨 PHASE 3: GENERATING COMPARISON PLOTS"
echo "----------------------------------------"

# Generate plots comparing router vs vLLM
ROUTER_RESULT=$(find "$OUTPUT_DIR" -name "*router*auto*" -type d | head -1)
VLLM_RESULT=$(find "$OUTPUT_DIR" -name "*vllm*gpt-oss*" -type d | head -1)

if [ -n "$ROUTER_RESULT" ] && [ -f "$ROUTER_RESULT/summary.json" ] && [ -n "$VLLM_RESULT" ] && [ -f "$VLLM_RESULT/summary.json" ]; then
    echo "Creating comparison plots (router plotted first for visibility)..."
    
    # Create plots directory
    PLOTS_DIR="$OUTPUT_DIR/plots"
    mkdir -p "$PLOTS_DIR"
    
    # Generate vLLM plots with router overlay (router plotted first)
    python3 -m vllm_semantic_router_bench.bench_plot \
        --summary "$VLLM_RESULT/summary.json" \
        --router-summary "$ROUTER_RESULT/summary.json" \
        --out-dir "$PLOTS_DIR" \
        --metrics accuracy avg_response_time avg_total_tokens \
        --font-scale 1.4 \
        --dpi 300
    
    echo "✅ Plots generated in: $PLOTS_DIR"
    echo "   - bench_plot_accuracy.png (+ PDF)"
    echo "   - bench_plot_avg_response_time.png (+ PDF)" 
    echo "   - bench_plot_avg_total_tokens.png (+ PDF)"
    echo "   📊 Router trend lines plotted first to remain visible even with overlapping dots"
else
    echo "⚠️  Skipping plots - missing result files"
fi

echo ""
echo "📊 BENCHMARK COMPLETED!"
echo "======================="
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Display quick summary if results exist
echo "📈 QUICK SUMMARY:"
echo "-----------------"

# Find and display router results
ROUTER_RESULT=$(find "$OUTPUT_DIR" -name "*router*auto*" -type d | head -1)
if [ -n "$ROUTER_RESULT" ] && [ -f "$ROUTER_RESULT/summary.json" ]; then
    echo "🔀 Router (via Envoy):"
    python3 -c "
import json, sys
try:
    with open('$ROUTER_RESULT/summary.json') as f:
        data = json.load(f)
    print(f\"  Accuracy: {data.get('overall_accuracy', 0):.3f}\")
    print(f\"  Avg Latency: {data.get('avg_response_time', 0):.2f}s\")
    print(f\"  Avg Tokens: {data.get('avg_total_tokens', 0):.0f}\")
    print(f\"  Questions: {data.get('successful_queries', 0)}/{data.get('total_questions', 0)}\")
except Exception as e:
    print(f\"  Error reading router results: {e}\")
"
fi

# Find and display vLLM results
VLLM_RESULT=$(find "$OUTPUT_DIR" -name "*vllm*gpt-oss*" -type d | head -1)
if [ -n "$VLLM_RESULT" ] && [ -f "$VLLM_RESULT/summary.json" ]; then
    echo "🎯 Direct vLLM:"
    python3 -c "
import json, sys
try:
    with open('$VLLM_RESULT/summary.json') as f:
        data = json.load(f)
    print(f\"  Accuracy: {data.get('overall_accuracy', 0):.3f}\")
    print(f\"  Avg Latency: {data.get('avg_response_time', 0):.2f}s\")
    print(f\"  Avg Tokens: {data.get('avg_total_tokens', 0):.0f}\")
    print(f\"  Questions: {data.get('successful_queries', 0)}/{data.get('total_questions', 0)}\")
    
    # Show breakdown by mode if available
    by_mode = data.get('by_mode', {})
    if by_mode:
        print(\"  Mode Breakdown:\")
        for mode, metrics in by_mode.items():
            if 'accuracy' in metrics:
                print(f\"    {mode}: {metrics['accuracy']:.3f} acc, {metrics.get('avg_response_time', 0):.2f}s\")
except Exception as e:
    print(f\"  Error reading vLLM results: {e}\")
"
fi

echo ""
echo "🔍 DETAILED ANALYSIS:"
echo "--------------------"
echo "- Router results: $ROUTER_RESULT"
echo "- vLLM results: $VLLM_RESULT"
echo "- Comparison plots: $OUTPUT_DIR/plots/"
echo "- Compare CSV files for detailed question-by-question analysis"
echo "- Check summary.json files for comprehensive metrics"
echo ""

echo "📊 VISUALIZATION FILES:"
echo "----------------------"
if [ -d "$OUTPUT_DIR/plots" ]; then
    echo "- Accuracy comparison: $OUTPUT_DIR/plots/bench_plot_accuracy.png"
    echo "- Response time comparison: $OUTPUT_DIR/plots/bench_plot_avg_response_time.png"
    echo "- Token usage comparison: $OUTPUT_DIR/plots/bench_plot_avg_total_tokens.png"
    echo "- PDF versions also available in same directory"
else
    echo "- No plots generated (check for errors above)"
fi
echo ""

echo "✅ Benchmark comparison complete!"
echo "Run with different datasets: $0 mmlu 10"
echo "Run with different datasets: $0 arc-challenge 3"
