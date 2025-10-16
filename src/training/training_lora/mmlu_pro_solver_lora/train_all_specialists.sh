#!/bin/bash
# Batch training script for all MMLU-Pro specialist models
# This will train 6 specialized Qwen3-0.6B models sequentially

set -e  # Exit on error

echo "======================================================================"
echo "MMLU-Pro Specialist Training Pipeline"
echo "======================================================================"
echo ""
echo "This script will train 6 specialized models:"
echo "  1. MathReasoner (math, physics, engineering)"
echo "  2. ScienceExpert (biology, chemistry, computer science)"
echo "  3. HumanitiesScholar (history, philosophy)"
echo "  4. SocialScientist (psychology, economics, business)"
echo "  5. LegalExpert (law)"
echo "  6. Generalist (health, other)"
echo ""
echo "Estimated total training time: 12-18 hours on RTX 3090"
echo "======================================================================"
echo ""

# Configuration
BASE_MODEL="Qwen/Qwen3-0.6B"
EPOCHS=5
BATCH_SIZE=2
LORA_RANK=32
LEARNING_RATE=2e-4
MAX_SAMPLES=200
GPU_ID=${1:-0}  # Default to GPU 0, or use first argument

echo "Configuration:"
echo "  Base Model: $BASE_MODEL"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  LoRA Rank: $LORA_RANK"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Max Samples/Category: $MAX_SAMPLES"
echo "  GPU ID: $GPU_ID"
echo ""

# Create logs directory
mkdir -p training_logs

# Function to train a model
train_model() {
    local MODEL_TYPE=$1
    local CUSTOM_EPOCHS=${2:-$EPOCHS}
    local CUSTOM_SAMPLES=${3:-$MAX_SAMPLES}
    local CUSTOM_LR=${4:-$LEARNING_RATE}
    
    echo ""
    echo "======================================================================"
    echo "Training: $MODEL_TYPE"
    echo "======================================================================"
    echo "  Epochs: $CUSTOM_EPOCHS"
    echo "  Samples/Category: $CUSTOM_SAMPLES"
    echo "  Learning Rate: $CUSTOM_LR"
    echo ""
    
    LOG_FILE="training_logs/${MODEL_TYPE}_$(date +%Y%m%d_%H%M%S).log"
    
    python ft_qwen3_mmlu_solver_lora.py \
        --mode train \
        --model "$BASE_MODEL" \
        --model-type "$MODEL_TYPE" \
        --epochs "$CUSTOM_EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lora-rank "$LORA_RANK" \
        --learning-rate "$CUSTOM_LR" \
        --max-samples-per-category "$CUSTOM_SAMPLES" \
        --gpu-id "$GPU_ID" \
        2>&1 | tee "$LOG_FILE"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Successfully trained $MODEL_TYPE"
        echo "  Log: $LOG_FILE"
        echo ""
    else
        echo ""
        echo "✗ Error training $MODEL_TYPE"
        echo "  Check log: $LOG_FILE"
        echo ""
        exit 1
    fi
}

# Start time
START_TIME=$(date +%s)

# 1. Train Math Reasoner (highest priority)
train_model "math-reasoner" 5 200 "2e-4"

# 2. Train Science Expert
train_model "science-expert" 5 200 "2e-4"

# 3. Train Humanities Scholar (more epochs, more samples due to smaller category count)
train_model "humanities" 6 250 "1.5e-4"

# 4. Train Social Scientist
train_model "social-sciences" 5 200 "1.5e-4"

# 5. Train Legal Expert (specialized single-category, more epochs)
train_model "law" 8 300 "1.5e-4"

# 6. Train Generalist
train_model "generalist" 5 200 "2e-4"

# End time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo ""
echo "======================================================================"
echo "Training Complete!"
echo "======================================================================"
echo ""
echo "Total training time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Trained models:"
echo "  1. qwen3_mmlu_math-reasoner_r32/"
echo "  2. qwen3_mmlu_science-expert_r32/"
echo "  3. qwen3_mmlu_humanities_r32/"
echo "  4. qwen3_mmlu_social-sciences_r32/"
echo "  5. qwen3_mmlu_law_r32/"
echo "  6. qwen3_mmlu_generalist_r32/"
echo ""
echo "Training logs saved in: training_logs/"
echo ""
echo "Next steps:"
echo "  1. Test each model:"
echo "     python ft_qwen3_mmlu_solver_lora.py --mode test --model-path qwen3_mmlu_math-reasoner_r32"
echo ""
echo "  2. Build a router system to combine all specialists"
echo ""
echo "  3. Evaluate on full MMLU-Pro test set"
echo ""
echo "======================================================================"

