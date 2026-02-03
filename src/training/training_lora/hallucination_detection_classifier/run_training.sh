#!/bin/bash
# =============================================================================
# Hallucination Detection Training Script
# =============================================================================
# Uses 32K ModernBERT as described in TRAINING_32K.md
# Best Configuration: RAGTruth + DART + E2E = 76.56% F1
# =============================================================================

set -e

# Get script directory (works even when called from different locations)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

# Configuration (can be overridden via environment variables)
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm:latest}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/haldetect-32k}"

# Model settings (from TRAINING_32K.md)
MODEL_NAME="${MODEL_NAME:-llm-semantic-router/modernbert-base-32k}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
EPOCHS="${EPOCHS:-6}"
EARLY_STOPPING="${EARLY_STOPPING:-4}"

# Source data paths (relative to project root)
RAGTRUTH_PATH="${RAGTRUTH_PATH:-${PROJECT_ROOT}/data/ragtruth/ragtruth_data.json}"
DART_PATH="${DART_PATH:-${PROJECT_ROOT}/data/dart_spans/dart_spans.json}"
E2E_PATH="${E2E_PATH:-${PROJECT_ROOT}/data/e2e_spans/e2e_spans.json}"

# GPU settings
GPU_DEVICES="${HIP_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"

echo "============================================================"
echo "HALLUCINATION DETECTION TRAINING PIPELINE"
echo "============================================================"
echo "Model: ${MODEL_NAME}"
echo "Max Length: ${MAX_LENGTH}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LEARNING_RATE}"
echo "Epochs: ${EPOCHS}"
echo "GPU Devices: ${GPU_DEVICES}"
echo "Script Dir: ${SCRIPT_DIR}"
echo "Project Root: ${PROJECT_ROOT}"
echo "============================================================"

# Step 1: Prepare Data
echo ""
echo "[Step 1/2] Preparing data..."
echo "------------------------------------------------------------"

docker run --rm \
    -v "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    -w "${SCRIPT_DIR}" \
    "${DOCKER_IMAGE}" \
    python3 prepare_data.py \
        --ragtruth-path "${RAGTRUTH_PATH}" \
        --dart-path "${DART_PATH}" \
        --e2e-path "${E2E_PATH}" \
        --output-dir "${DATA_DIR}"

# Step 2: Fine-tune Model
echo ""
echo "[Step 2/2] Fine-tuning model..."
echo "------------------------------------------------------------"

docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --security-opt seccomp=unconfined \
    -e HIP_VISIBLE_DEVICES="${GPU_DEVICES}" \
    -e CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" \
    -v "${PROJECT_ROOT}:${PROJECT_ROOT}" \
    -w "${SCRIPT_DIR}" \
    "${DOCKER_IMAGE}" \
    python3 finetune.py \
        --train-path "${DATA_DIR}/train.json" \
        --dev-path "${DATA_DIR}/dev.json" \
        --test-path "${DATA_DIR}/test.json" \
        --output-dir "${OUTPUT_DIR}" \
        --model-name "${MODEL_NAME}" \
        --max-length "${MAX_LENGTH}" \
        --batch-size "${BATCH_SIZE}" \
        --learning-rate "${LEARNING_RATE}" \
        --epochs "${EPOCHS}" \
        --early-stopping-patience "${EARLY_STOPPING}"

echo ""
echo "============================================================"
echo "TRAINING COMPLETE"
echo "============================================================"
echo "Model saved to: ${OUTPUT_DIR}"
echo "============================================================"
