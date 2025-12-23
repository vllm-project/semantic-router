#!/bin/bash
set -e

echo "=========================================="
echo "Installing dependencies..."
echo "=========================================="
pip install transformers datasets scikit-learn tqdm accelerate -q

echo ""
echo "=========================================="
echo "Training Stage 1 and Stage 2 in PARALLEL"
echo "=========================================="

# Train Stage 1 (Sentinel) in background
echo "Starting Stage 1 training..."
python train_sentinel.py \
    --train-data data/augmented_v3/stage1_train.json \
    --dev-data data/augmented_v3/stage1_dev.json \
    --output-dir output/sentinel_augmented_v3 \
    --model-name answerdotai/ModernBERT-base \
    --batch-size 32 \
    --epochs 5 \
    --learning-rate 3e-5 \
    --max-length 512 \
    --seed 42 &
STAGE1_PID=$!

# Small delay to avoid GPU contention at startup
sleep 10

# Train Stage 2 (ToolCallVerifier) in background
echo "Starting Stage 2 training..."
python train_tool_verifier.py \
    --train-data data/augmented_v3/stage2_train.json \
    --dev-data data/augmented_v3/stage2_dev.json \
    --output-dir output/verifier_augmented_v3 \
    --model-name answerdotai/ModernBERT-base \
    --batch-size 16 \
    --epochs 5 \
    --learning-rate 3e-5 \
    --max-length 1024 \
    --seed 42 &
STAGE2_PID=$!

echo ""
echo "Both training jobs started!"
echo "Stage 1 PID: $STAGE1_PID"
echo "Stage 2 PID: $STAGE2_PID"
echo ""

# Wait for both to complete
wait $STAGE1_PID
STAGE1_EXIT=$?
echo "Stage 1 completed with exit code: $STAGE1_EXIT"

wait $STAGE2_PID
STAGE2_EXIT=$?
echo "Stage 2 completed with exit code: $STAGE2_EXIT"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Stage 1 model: output/sentinel_augmented_v3"
echo "Stage 2 model: output/verifier_augmented_v3"

if [ $STAGE1_EXIT -eq 0 ] && [ $STAGE2_EXIT -eq 0 ]; then
    echo "✅ Both models trained successfully!"
    exit 0
else
    echo "❌ Training failed (Stage1: $STAGE1_EXIT, Stage2: $STAGE2_EXIT)"
    exit 1
fi

