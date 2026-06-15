#!/bin/bash
# Router-R1 GPU Training Script
# 
# Requirements:
#   - NVIDIA GPU with 24GB+ VRAM (A100 recommended)
#   - CUDA 11.8+
#   - Python 3.10+
#
# Usage:
#   ./scripts/train_router_r1_gpu.sh [OPTIONS]
#
# Options:
#   --epochs NUM      Number of training epochs (default: 10)
#   --batch-size NUM  Batch size (default: 8)
#   --model NAME      Base model (default: microsoft/Phi-3-mini-4k-instruct)
#   --output DIR      Output directory

set -e

# Default values
EPOCHS=10
BATCH_SIZE=8
BASE_MODEL="microsoft/Phi-3-mini-4k-instruct"
OUTPUT_DIR="./checkpoints/router_r1"
CONFIG_PATH="./configs/router_r1_config.yaml"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Print configuration
echo "=============================================="
echo "Router-R1 GPU Training"
echo "=============================================="
echo "Base Model:    $BASE_MODEL"
echo "Epochs:        $EPOCHS"
echo "Batch Size:    $BATCH_SIZE"
echo "Output Dir:    $OUTPUT_DIR"
echo "Config:        $CONFIG_PATH"
echo "=============================================="

# Check GPU
if ! nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA GPU not detected. This script requires GPU."
    exit 1
fi

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Check Python dependencies
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Download data if needed
if [ ! -f "./data/train_routing.json" ]; then
    echo "Training data not found. Generating synthetic data..."
    python3 -c "
from train_router_r1 import RoutingDataset, RouterR1Config
from transformers import AutoTokenizer
config = RouterR1Config()
tokenizer = AutoTokenizer.from_pretrained(config.base_model)
tokenizer.pad_token = tokenizer.eos_token
dataset = RoutingDataset('./data/train_routing.json', tokenizer)
print(f'Generated {len(dataset)} synthetic training examples')
"
fi

# Start training
echo ""
echo "Starting Router-R1 training..."
echo ""

python3 train_router_r1.py \
    --config "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE"

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo "Model saved to: $OUTPUT_DIR"
echo ""

# Verify output
if [ -d "$OUTPUT_DIR/final_model" ]; then
    echo "Final model checkpoint: $OUTPUT_DIR/final_model"
    ls -la "$OUTPUT_DIR/final_model"
fi

echo ""
echo "To use this model in VSR, add to your config:"
echo ""
echo "decisions:"
echo "  - name: intelligent_routing"
echo "    algorithm:"
echo "      type: \"rl_driven\""
echo "      rl_driven:"
echo "        router_model_path: \"$OUTPUT_DIR/final_model\""
