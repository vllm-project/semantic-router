#!/bin/bash
# GPU Training Script for mmBERT LoRA Adapters using ROCm Docker
# Optimized hyperparameters for high accuracy

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  mmBERT GPU Training (Optimized)     ${NC}"
echo -e "${BLUE}========================================${NC}"

# OPTIMIZED Training configuration for high accuracy
MODEL="mmbert-base"
EPOCHS=10             # More epochs for better convergence
BATCH_SIZE=64         # Balanced batch size (not too large)
LORA_RANK=32          # Higher rank for more capacity
LORA_ALPHA=64         # 2x rank
MAX_SAMPLES=10000     # Large dataset
LEARNING_RATE=2e-5    # Lower LR for stability

echo -e "${YELLOW}Optimized Configuration:${NC}"
echo "  Model: $MODEL"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  LoRA Rank: $LORA_RANK (higher capacity)"
echo "  LoRA Alpha: $LORA_ALPHA"
echo "  Max Samples: $MAX_SAMPLES"
echo "  Learning Rate: $LEARNING_RATE (lower for stability)"
echo ""

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -q peft accelerate datasets transformers scikit-learn tqdm requests

# Verify GPU
echo -e "${YELLOW}Checking GPU...${NC}"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB' if torch.cuda.is_available() else '')"

# Train each task
train_task() {
    local task_name=$1
    local script_path=$2
    
    echo -e "${BLUE}========================================${NC}"
    echo -e "${GREEN}Training: $task_name${NC}"
    echo -e "${BLUE}========================================${NC}"
    
    start_time=$(date +%s)
    
    python "$script_path" \
        --mode train \
        --model "$MODEL" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lora-rank "$LORA_RANK" \
        --lora-alpha "$LORA_ALPHA" \
        --max-samples "$MAX_SAMPLES" \
        --learning-rate "$LEARNING_RATE"
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo -e "${GREEN}✓ $task_name completed in ${duration}s${NC}"
}

cd /workspace/training_lora

# 1. Intent Classification
train_task "Intent Classifier" "classifier_model_fine_tuning_lora/ft_linear_lora.py"

# 2. Fact-Check Classification  
train_task "Fact-Check Classifier" "fact_check_fine_tuning_lora/fact_check_bert_finetuning_lora.py"

# 3. PII Detection
train_task "PII Detector" "pii_model_fine_tuning_lora/pii_bert_finetuning_lora.py"

# 4. Jailbreak Detection
train_task "Jailbreak Detector" "prompt_guard_fine_tuning_lora/jailbreak_bert_finetuning_lora.py"

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}  All Training Complete!              ${NC}"
echo -e "${BLUE}========================================${NC}"

# List trained models
echo -e "${YELLOW}Trained Models:${NC}"
for dir in lora_*; do [ -d "$dir" ] && ls -ld "$dir"; done || true

# Copy models to output
cp -r lora_*mmbert* /workspace/models/ 2>/dev/null || true
echo -e "${GREEN}Models copied to /workspace/models/${NC}"
