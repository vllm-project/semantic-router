#!/bin/bash
# =============================================================================
# Train mmBERT-32K models using ROCm GPU inside Docker container
# =============================================================================

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration
DOCKER_IMAGE="${DOCKER_IMAGE:-rocm/vllm:v0.14.0_amd_dev}"
CONTAINER_WORKDIR="/workspace"
MODELS_DIR="${MODELS_DIR:-models/mmbert32k}"

# Training parameters (can be overridden)
# Updated 2026-02-02 with validated hyperparameters
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
TRAIN_LR="${TRAIN_LR:-2e-5}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
MAX_SAMPLES="${MAX_SAMPLES:-10000}"

# Feedback-specific parameters (higher capacity for 4-class)
FEEDBACK_EPOCHS="${FEEDBACK_EPOCHS:-10}"
FEEDBACK_LR="${FEEDBACK_LR:-2e-5}"
FEEDBACK_LORA_RANK="${FEEDBACK_LORA_RANK:-64}"
FEEDBACK_LORA_ALPHA="${FEEDBACK_LORA_ALPHA:-128}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  mmBERT-32K GPU Training (ROCm)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Docker image: ${YELLOW}${DOCKER_IMAGE}${NC}"
echo -e "General: Epochs=${TRAIN_EPOCHS}, Batch=${TRAIN_BATCH_SIZE}, LR=${TRAIN_LR}"
echo -e "General LoRA: rank=${LORA_RANK}, alpha=${LORA_ALPHA}"
echo -e "Feedback: Epochs=${FEEDBACK_EPOCHS}, rank=${FEEDBACK_LORA_RANK}, alpha=${FEEDBACK_LORA_ALPHA}"
echo ""

# Verify project root has expected structure
if [ ! -f "${PROJECT_ROOT}/Makefile" ] || [ ! -d "${PROJECT_ROOT}/src/training" ]; then
    echo -e "${RED}Error: Cannot find project root. Expected Makefile and src/training/ in ${PROJECT_ROOT}${NC}"
    exit 1
fi

# Change to project root
cd "${PROJECT_ROOT}"

# Create models directory
mkdir -p "${MODELS_DIR}"

# Build the training command
TRAIN_SCRIPT=$(cat << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
"""
mmBERT-32K Training Script for GPU
Trains all 5 models with LoRA and merges them.
"""

import os
import sys
import subprocess
import time

# Get working directory (mounted project root)
WORKDIR = os.getcwd()

# Add src to path
sys.path.insert(0, os.path.join(WORKDIR, 'src'))

def install_dependencies():
    """Install required packages."""
    print("\n📦 Installing dependencies...")
    packages = [
        "transformers>=4.40.0",
        "peft>=0.10.0", 
        "datasets>=2.18.0",
        "accelerate>=0.27.0",
        "scikit-learn>=1.4.0",
        "sentencepiece",
        "safetensors",
    ]
    for pkg in packages:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)
    print("✅ Dependencies installed")

def train_model(name, script_path, extra_args=None):
    """Train a single model."""
    print(f"\n{'='*60}")
    print(f"🚀 Training: {name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Use relative path from working directory
    full_script_path = os.path.join(WORKDIR, script_path)
    cmd = [sys.executable, full_script_path]
    if extra_args:
        cmd.extend(extra_args)
    
    result = subprocess.run(cmd, cwd=WORKDIR)
    
    elapsed = time.time() - start_time
    if result.returncode == 0:
        print(f"✅ {name} completed in {elapsed/60:.1f} minutes")
    else:
        print(f"❌ {name} failed (exit code: {result.returncode})")
    
    return result.returncode == 0

def main():
    # Get environment variables
    epochs = os.environ.get('TRAIN_EPOCHS', '5')
    batch_size = os.environ.get('TRAIN_BATCH_SIZE', '16')
    lr = os.environ.get('TRAIN_LR', '2e-5')
    lora_rank = os.environ.get('LORA_RANK', '32')
    lora_alpha = os.environ.get('LORA_ALPHA', '64')
    max_samples = os.environ.get('MAX_SAMPLES', '10000')
    models_dir = os.environ.get('MODELS_DIR', 'models/mmbert32k')
    
    # Feedback-specific (higher capacity for 4-class classification)
    feedback_epochs = os.environ.get('FEEDBACK_EPOCHS', '10')
    feedback_lr = os.environ.get('FEEDBACK_LR', '2e-5')
    feedback_lora_rank = os.environ.get('FEEDBACK_LORA_RANK', '64')
    feedback_lora_alpha = os.environ.get('FEEDBACK_LORA_ALPHA', '128')
    
    # Check GPU
    import torch
    print(f"\n🖥️  GPU Status:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA/ROCm available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    install_dependencies()
    
    os.makedirs(models_dir, exist_ok=True)
    
    results = {}
    total_start = time.time()
    
    # 1. Feedback Detector (uses higher rank for 4-class classification)
    print(f"\n📊 Feedback Detector: rank={feedback_lora_rank}, alpha={feedback_lora_alpha}, epochs={feedback_epochs}")
    results['feedback'] = train_model(
        "Feedback Detector",
        "src/training/modernbert_dissat_pipeline/train_feedback_detector.py",
        [
            "--model_name", "llm-semantic-router/mmbert-32k-yarn",
            "--output_dir", f"{models_dir}/feedback-detector",
            "--epochs", feedback_epochs,
            "--batch_size", batch_size,
            "--lr", feedback_lr,
            "--use_lora",
            "--lora_rank", feedback_lora_rank,
            "--lora_alpha", feedback_lora_alpha,
            "--merge_lora",
        ]
    )
    
    # 2. Intent Classifier
    results['intent'] = train_model(
        "Intent Classifier",
        "src/training/training_lora/classifier_model_fine_tuning_lora/ft_linear_lora.py",
        [
            "--mode", "train",
            "--model", "mmbert-32k",
            "--lora-rank", lora_rank,
            "--lora-alpha", lora_alpha,
            "--epochs", epochs,
            "--batch-size", batch_size,
            "--learning-rate", lr,
            "--max-samples", max_samples,
        ]
    )
    
    # 3. PII Detector
    results['pii'] = train_model(
        "PII Detector",
        "src/training/training_lora/pii_model_fine_tuning_lora/pii_bert_finetuning_lora.py",
        [
            "--mode", "train",
            "--model", "mmbert-32k",
            "--lora-rank", lora_rank,
            "--lora-alpha", lora_alpha,
            "--epochs", epochs,
            "--batch-size", batch_size,
            "--learning-rate", lr,
        ]
    )
    
    # 4. Jailbreak Detector
    results['jailbreak'] = train_model(
        "Jailbreak Detector",
        "src/training/training_lora/prompt_guard_fine_tuning_lora/jailbreak_bert_finetuning_lora.py",
        [
            "--mode", "train",
            "--model", "mmbert-32k",
            "--lora-rank", lora_rank,
            "--lora-alpha", lora_alpha,
            "--epochs", epochs,
            "--batch-size", batch_size,
            "--learning-rate", lr,
        ]
    )
    
    # 5. Fact Check Classifier
    results['factcheck'] = train_model(
        "Fact Check Classifier",
        "src/training/training_lora/fact_check_fine_tuning_lora/fact_check_bert_finetuning_lora.py",
        [
            "--mode", "train",
            "--model", "mmbert-32k",
            "--lora-rank", "16",  # Higher rank for fact-check
            "--lora-alpha", "32",
            "--epochs", epochs,
            "--batch-size", batch_size,
            "--learning-rate", lr,
            "--max-samples", max_samples,
        ]
    )
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print("")
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    success_count = sum(results.values())
    print(f"\n{success_count}/{len(results)} models trained successfully")
    
    if success_count == len(results):
        print("\n🎉 All models trained! Output in:", models_dir)
    
    return 0 if success_count == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())
PYTHON_SCRIPT
)

# Run training in Docker container
echo -e "${GREEN}Starting training in Docker container...${NC}"
echo -e "Project root: ${PROJECT_ROOT}"
echo ""

docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --shm-size=16g \
    -v "${PROJECT_ROOT}:${CONTAINER_WORKDIR}" \
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
    -w "${CONTAINER_WORKDIR}" \
    -e TRAIN_EPOCHS="${TRAIN_EPOCHS}" \
    -e TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
    -e TRAIN_LR="${TRAIN_LR}" \
    -e LORA_RANK="${LORA_RANK}" \
    -e LORA_ALPHA="${LORA_ALPHA}" \
    -e MAX_SAMPLES="${MAX_SAMPLES}" \
    -e MODELS_DIR="${MODELS_DIR}" \
    -e FEEDBACK_EPOCHS="${FEEDBACK_EPOCHS}" \
    -e FEEDBACK_LR="${FEEDBACK_LR}" \
    -e FEEDBACK_LORA_RANK="${FEEDBACK_LORA_RANK}" \
    -e FEEDBACK_LORA_ALPHA="${FEEDBACK_LORA_ALPHA}" \
    -e HF_HOME="/root/.cache/huggingface" \
    -e TRANSFORMERS_CACHE="/root/.cache/huggingface/hub" \
    "${DOCKER_IMAGE}" \
    python3 -c "${TRAIN_SCRIPT}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Training Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Models saved to: ${PROJECT_ROOT}/${MODELS_DIR}/"
ls -la "${PROJECT_ROOT}/${MODELS_DIR}/" 2>/dev/null || echo "(check output above for results)"
