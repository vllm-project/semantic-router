#!/bin/bash
set -e

echo "=========================================="
echo "Installing dependencies..."
echo "=========================================="
pip install transformers datasets scikit-learn tqdm accelerate -q

echo ""
echo "=========================================="
echo "Stage 1: Training Sentinel (ModernBERT)"
echo "=========================================="
python train_sentinel.py \
    --train-data data/fresh_v2/stage1_train.json \
    --dev-data data/fresh_v2/stage1_dev.json \
    --output-dir output/sentinel_fresh_v2 \
    --model-name answerdotai/ModernBERT-base \
    --batch-size 32 \
    --epochs 5 \
    --learning-rate 3e-5 \
    --max-length 512 \
    --seed 42

echo ""
echo "=========================================="
echo "Stage 2: Training ToolCallVerifier (ModernBERT)"
echo "=========================================="
python train_tool_verifier.py \
    --train-data data/fresh_v2/stage2_train.json \
    --dev-data data/fresh_v2/stage2_dev.json \
    --output-dir output/verifier_fresh_v2 \
    --model-name answerdotai/ModernBERT-base \
    --batch-size 16 \
    --epochs 5 \
    --learning-rate 3e-5 \
    --max-length 1024 \
    --seed 42

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Stage 1 model: output/sentinel_fresh_v2"
echo "Stage 2 model: output/verifier_fresh_v2"

