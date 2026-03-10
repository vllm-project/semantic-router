#!/bin/bash
# GMTRouter GPU Training Script
# 
# Requirements:
#   - NVIDIA GPU with 16GB+ VRAM
#   - CUDA 11.8+
#   - PyTorch Geometric
#
# Usage:
#   ./scripts/train_gmtrouter_gpu.sh [OPTIONS]

set -e

# Default values
EPOCHS=50
BATCH_SIZE=32
OUTPUT_DIR="./checkpoints/gmtrouter"
CONFIG_PATH="./configs/gmtrouter_config.yaml"

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
echo "GMTRouter GPU Training"
echo "=============================================="
echo "Epochs:        $EPOCHS"
echo "Batch Size:    $BATCH_SIZE"
echo "Output Dir:    $OUTPUT_DIR"
echo "Config:        $CONFIG_PATH"
echo "=============================================="

# Check GPU
if ! nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA GPU not detected."
    exit 1
fi

echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Check Python dependencies
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Check PyTorch Geometric (optional, will use simplified version if not available)
python3 -c "
try:
    import torch_geometric
    print(f'PyTorch Geometric version: {torch_geometric.__version__}')
except ImportError:
    print('PyTorch Geometric not installed. Using simplified graph model.')
"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start training
echo ""
echo "Starting GMTRouter training..."
echo ""

python3 train_gmtrouter.py \
    --config "$CONFIG_PATH" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo "Model saved to: $OUTPUT_DIR"
