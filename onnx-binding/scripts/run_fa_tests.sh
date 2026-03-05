#!/usr/bin/env bash
# Run Flash Attention integration tests for onnx-binding.
#
# Usage (inside Docker container):
#   ./scripts/run_fa_tests.sh [--download] [--test-filter PATTERN]
#
# Each test is run as a separate process to avoid GPU OOM from accumulating
# ONNX Runtime sessions (ORT doesn't release GPU memory until process exit).
set -euo pipefail

FA_MODELS_DIR="${FA_MODELS_DIR:-/models}"
DOWNLOAD=false
TEST_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --download)
            DOWNLOAD=true; shift ;;
        --test-filter)
            TEST_FILTER="$2"; shift 2 ;;
        *)
            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "  Flash Attention Integration Tests"
echo "============================================================"
echo "  FA_MODELS_DIR:         $FA_MODELS_DIR"
echo "  ORT_CK_FLASH_ATTN_LIB: ${ORT_CK_FLASH_ATTN_LIB:-<not set>}"
echo "  ORT_DYLIB_PATH:        ${ORT_DYLIB_PATH:-<not set>}"
echo "============================================================"

if [ -z "${ORT_CK_FLASH_ATTN_LIB:-}" ]; then
    echo "ERROR: ORT_CK_FLASH_ATTN_LIB not set"
    exit 1
fi
if [ ! -f "$ORT_CK_FLASH_ATTN_LIB" ]; then
    echo "ERROR: FA library not found: $ORT_CK_FLASH_ATTN_LIB"
    exit 1
fi

if [ "$DOWNLOAD" = true ]; then
    echo ""
    echo "Downloading FA models..."
    python3 /workspace/scripts/download_fa_models.py
fi

echo ""
echo "Checking model directories..."
for model in \
    mmbert-embed-32k-2d-matryoshka \
    mmbert32k-intent-classifier-merged \
    mmbert32k-jailbreak-detector-merged \
    mmbert32k-pii-detector-merged \
    mmbert32k-factcheck-classifier-merged \
    mmbert32k-feedback-detector-merged; do
    dir="$FA_MODELS_DIR/$model"
    if [ -d "$dir" ]; then
        fa_file=$(find "$dir" -name "model_fa_fp16.onnx" 2>/dev/null | head -1)
        if [ -n "$fa_file" ]; then
            echo "  OK: $model ($(du -sh "$fa_file" | cut -f1))"
        else
            echo "  WARN: $model — no model_fa_fp16.onnx"
        fi
    else
        echo "  MISSING: $model"
    fi
done

cd /workspace
export LD_LIBRARY_PATH="/workspace/target/release:${LD_LIBRARY_PATH:-}"
export FA_MODELS_DIR

# Run each top-level test in its own process to avoid GPU OOM accumulation.
# ORT sessions hold GPU memory until the process exits.
TESTS=(
    "TestFA_Embedding"
    "TestFA_Classifier_Intent"
    "TestFA_Classifier_Jailbreak"
    "TestFA_Classifier_Factcheck"
    "TestFA_Classifier_Feedback"
    "TestFA_Classifier_PII"
)

if [ -n "$TEST_FILTER" ]; then
    TESTS=("$TEST_FILTER")
fi

PASS=0
FAIL=0

echo ""
echo "============================================================"
echo "  Running FA tests (each in isolated process)"
echo "============================================================"

for test_name in "${TESTS[@]}"; do
    echo ""
    echo "---- $test_name ----"
    if go test -tags fa -v -count=1 -timeout 300s -run "^${test_name}\$" . 2>&1; then
        PASS=$((PASS+1))
    else
        FAIL=$((FAIL+1))
        echo "  FAILED"
    fi
done

echo ""
echo "============================================================"
echo "  FA Test Summary: PASS=$PASS  FAIL=$FAIL"
echo "============================================================"

[ "$FAIL" -eq 0 ] || exit 1
