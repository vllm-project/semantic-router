#!/bin/bash
#
# Reasoning Mode Evaluation Script
# Issue #42: [v0.1]Bench: Reasoning mode evaluation
#
# Compares standard vs reasoning mode using:
# - Response correctness on MMLU(-Pro) and non-MMLU test sets
# - Token usage (completion_tokens/prompt_tokens ratio)
# - Response time per output token
#
# Usage:
#   ./reasoning_mode_eval.sh [options]
#
# Environment Variables:
#   VLLM_ENDPOINT - vLLM endpoint URL (default: http://127.0.0.1:8000/v1)
#   VLLM_API_KEY  - API key for vLLM endpoint (default: 1234)
#   MODEL         - Model to evaluate (fetches from endpoint if not set)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default configuration
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:8000/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-1234}"
OUTPUT_DIR="${OUTPUT_DIR:-results/reasoning_mode_eval}"
SAMPLES="${SAMPLES:-10}"
CONCURRENT="${CONCURRENT:-1}"

# Default datasets: MMLU (primary) and non-MMLU (for comparison)
DATASETS="${DATASETS:-mmlu gpqa truthfulqa}"

echo "=============================================="
echo "🧠 Reasoning Mode Evaluation (Issue #42)"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Endpoint:    ${VLLM_ENDPOINT}"
echo "  Datasets:    ${DATASETS}"
echo "  Samples:     ${SAMPLES} per category"
echo "  Concurrent:  ${CONCURRENT} requests"
echo "  Output:      ${OUTPUT_DIR}"
echo ""

# Build command
CMD="python -m reasoning.reasoning_mode_eval \
    --datasets ${DATASETS} \
    --endpoint ${VLLM_ENDPOINT} \
    --api-key ${VLLM_API_KEY} \
    --samples-per-category ${SAMPLES} \
    --concurrent-requests ${CONCURRENT} \
    --output-dir ${OUTPUT_DIR}"

# Add model if specified
if [ -n "${MODEL}" ]; then
    CMD="${CMD} --model ${MODEL}"
    echo "  Model:       ${MODEL}"
fi

echo ""
echo "Running evaluation..."
echo ""

cd "${SCRIPT_DIR}"
eval "${CMD}"

echo ""
echo "=============================================="
echo "✅ Evaluation Complete"
echo "=============================================="
echo ""
echo "Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Key outputs:"
echo "  - reasoning_mode_eval_summary.json  (JSON summary)"
echo "  - REASONING_MODE_EVALUATION_REPORT.md (Markdown report)"
echo "  - plots/                            (Visualization plots)"
echo "  - <dataset>/detailed_results.csv   (Per-question results)"
echo ""

