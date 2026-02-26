#!/bin/bash
# Modality Routing Classifier Training Script
# Trains mmBERT-32K YaRN with LoRA for 3-class classification: AR / DIFFUSION / BOTH
#
# Usage:
#   # Basic training (template-based BOTH class)
#   bash run_training.sh
#
#   # With vLLM synthesis for BOTH class
#   VLLM_ENDPOINT=http://localhost:8000/v1 bash run_training.sh
#
#   # Custom configuration
#   MODEL=mmbert-base EPOCHS=10 BATCH_SIZE=64 bash run_training.sh

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}  Modality Routing Classifier Training     ${NC}"
echo -e "${BLUE}  (AR / DIFFUSION / BOTH)                  ${NC}"
echo -e "${BLUE}============================================${NC}"

# Configuration (override via environment variables)
MODEL="${MODEL:-mmbert-32k}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LORA_RANK="${LORA_RANK:-}"        # Empty = auto-scale based on data volume
LORA_ALPHA="${LORA_ALPHA:-}"      # Empty = 2x rank (auto)
MAX_SAMPLES="${MAX_SAMPLES:-6000}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"

# vLLM synthesis configuration
VLLM_ENDPOINT="${VLLM_ENDPOINT:-}"
VLLM_MODEL="${VLLM_MODEL:-}"
SYNTHESIZE_BOTH="${SYNTHESIZE_BOTH:-0}"

# Auto-detect: if VLLM_ENDPOINT is set but SYNTHESIZE_BOTH is 0, default to 2000
if [ -n "$VLLM_ENDPOINT" ] && [ "$SYNTHESIZE_BOTH" -eq 0 ]; then
    SYNTHESIZE_BOTH=2000
    echo -e "${YELLOW}vLLM endpoint detected, auto-setting SYNTHESIZE_BOTH=2000${NC}"
fi

echo -e "${YELLOW}Configuration:${NC}"
echo "  Model: $MODEL"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  LoRA Rank: ${LORA_RANK:-auto (scales with data volume)}"
echo "  LoRA Alpha: ${LORA_ALPHA:-auto (2x rank)}"
echo "  Max Samples: $MAX_SAMPLES"
echo "  Learning Rate: $LEARNING_RATE"
if [ -n "$VLLM_ENDPOINT" ]; then
    echo "  vLLM Endpoint: $VLLM_ENDPOINT"
    echo "  vLLM Model: ${VLLM_MODEL:-auto-detect}"
    echo "  Synthesize BOTH: $SYNTHESIZE_BOTH"
else
    echo "  vLLM Synthesis: disabled (using templates)"
fi
echo ""

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -q peft accelerate datasets transformers scikit-learn tqdm requests

# Verify GPU
echo -e "${YELLOW}Checking GPU...${NC}"
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
else:
    print('No GPU detected - training will use CPU (slower)')
"

# Navigate to script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}Training: Modality Router${NC}"
echo -e "${BLUE}============================================${NC}"

start_time=$(date +%s)

# Build command
CMD="python modality_routing_bert_finetuning_lora.py \
    --mode train \
    --model $MODEL \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --max-samples $MAX_SAMPLES \
    --learning-rate $LEARNING_RATE"

# Add LoRA rank/alpha only if explicitly set (otherwise auto-scales)
if [ -n "$LORA_RANK" ]; then
    CMD="$CMD --lora-rank $LORA_RANK"
fi
if [ -n "$LORA_ALPHA" ]; then
    CMD="$CMD --lora-alpha $LORA_ALPHA"
fi

# Add vLLM options if configured
if [ -n "$VLLM_ENDPOINT" ]; then
    CMD="$CMD --vllm-endpoint $VLLM_ENDPOINT --synthesize-both $SYNTHESIZE_BOTH"
    if [ -n "$VLLM_MODEL" ]; then
        CMD="$CMD --vllm-model $VLLM_MODEL"
    fi
fi

# Run training
eval "$CMD"

end_time=$(date +%s)
duration=$((end_time - start_time))

echo -e "${GREEN}✓ Training completed in ${duration}s${NC}"

# Run inference test
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}Running inference test...${NC}"
echo -e "${BLUE}============================================${NC}"

# Find the output model directory (rank may be auto-selected)
MODEL_PATH=$(ls -d lora_modality_router_"${MODEL}"_r*_model 2>/dev/null | head -1)
if [ -n "$MODEL_PATH" ] && [ -d "$MODEL_PATH" ]; then
    echo -e "${GREEN}Found model at: $MODEL_PATH${NC}"
    python modality_routing_bert_finetuning_lora.py \
        --mode test \
        --model "$MODEL" \
        --model-path "$MODEL_PATH"
else
    echo -e "${YELLOW}Model directory not found, skipping inference test${NC}"
fi

echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}  Training Complete!                       ${NC}"
echo -e "${BLUE}============================================${NC}"

# List outputs
echo -e "${YELLOW}Output files:${NC}"
if [ -n "$MODEL_PATH" ]; then
    ls -la "$MODEL_PATH"/ 2>/dev/null || echo "  (no output directory found)"
else
    echo "  (no output directory found)"
fi
