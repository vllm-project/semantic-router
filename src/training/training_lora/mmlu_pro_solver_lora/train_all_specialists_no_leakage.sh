#!/bin/bash
#
# Batch Training Script for All MMLU-Pro Specialists (NO DATA LEAKAGE)
#
# This script trains all 6 specialized models using external datasets
# and tests them on MMLU-Pro as a held-out benchmark.
#
# Usage:
#   ./train_all_specialists_no_leakage.sh [GPU_ID] [SAMPLES_PER_DATASET] [EPOCHS]
#
# Examples:
#   ./train_all_specialists_no_leakage.sh 2 1000 5    # Full training on GPU 2
#   ./train_all_specialists_no_leakage.sh 3 100 2     # Quick test on GPU 3
#   ./train_all_specialists_no_leakage.sh             # Default: GPU 2, 1000 samples, 5 epochs

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default parameters (can be overridden by command line arguments)
GPU_ID=${1:-2}
SAMPLES_PER_DATASET=${2:-1000}
EPOCHS=${3:-5}
BATCH_SIZE=2
LORA_RANK=32

# Output directory
OUTPUT_BASE_DIR="models_no_leakage"
LOG_DIR="training_logs_no_leakage"

# Training script
TRAINING_SCRIPT="ft_qwen3_mmlu_solver_lora_no_leakage.py"

# ============================================================================
# DISPLAY BANNER
# ============================================================================

cat << 'EOF'

╔══════════════════════════════════════════════════════════════════╗
║     Batch Training - All MMLU-Pro Specialists (NO LEAKAGE)      ║
╚══════════════════════════════════════════════════════════════════╝

Training 6 specialized models using external datasets:
  1. Math Reasoner    (ARC)
  2. Science Expert   (ARC + OpenBookQA + SciQ)
  3. Social Sciences  (CommonsenseQA + StrategyQA)
  4. Humanities       (TruthfulQA)
  5. Law              (MMLU-train law only)
  6. Generalist       (ARC + CommonsenseQA + TruthfulQA)

Testing on: MMLU-Pro (held-out benchmark)
Data Leakage: ✅ NONE (completely separate datasets!)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EOF

echo "Configuration:"
echo "  GPU ID: $GPU_ID"
echo "  Samples per dataset: $SAMPLES_PER_DATASET"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  LoRA rank: $LORA_RANK"
echo "  Output directory: $OUTPUT_BASE_DIR/"
echo "  Log directory: $LOG_DIR/"
echo ""

# ============================================================================
# SETUP
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SETUP"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create directories
mkdir -p "$OUTPUT_BASE_DIR"
mkdir -p "$LOG_DIR"

# Check if training script exists
if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo "❌ ERROR: Training script not found: $TRAINING_SCRIPT"
    echo "   Make sure you're running this from: src/training/training_lora/mmlu_pro_solver_lora/"
    exit 1
fi
echo "✓ Training script found: $TRAINING_SCRIPT"
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.free,memory.total --format=csv,noheader,nounits | \
        awk '{printf "  GPU %s: %s - %.1f/%.1f GB free\n", $1, $2, $3/1024, $4/1024}'
    echo ""
else
    echo "⚠️  WARNING: nvidia-smi not found. Cannot check GPU status."
    echo ""
fi

# Estimate total training time
TOTAL_TIME_MIN=$((EPOCHS * SAMPLES_PER_DATASET / 50))  # Rough estimate
TOTAL_TIME_MAX=$((EPOCHS * SAMPLES_PER_DATASET / 20))
echo "Estimated total time: ${TOTAL_TIME_MIN}-${TOTAL_TIME_MAX} minutes (~$((TOTAL_TIME_MIN/60))-$((TOTAL_TIME_MAX/60)) hours)"
echo ""

# ============================================================================
# START TIMESTAMP
# ============================================================================

START_TIME=$(date +%s)
START_TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TRAINING STARTED: $START_TIMESTAMP"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

train_specialist() {
    local MODEL_TYPE=$1
    local MODEL_EPOCHS=$2
    local MODEL_SAMPLES=$3
    local DESCRIPTION=$4
    local TRAINING_DATA=$5
    local TEST_CATEGORIES=$6

    local MODEL_START_TIME=$(date +%s)
    local MODEL_START_TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")

    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║  Training: $MODEL_TYPE"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Description: $DESCRIPTION"
    echo "Training data: $TRAINING_DATA"
    echo "Test categories: $TEST_CATEGORIES"
    echo "Started: $MODEL_START_TIMESTAMP"
    echo ""

    local OUTPUT_DIR="$OUTPUT_BASE_DIR/${MODEL_TYPE}_r${LORA_RANK}_e${MODEL_EPOCHS}_s${MODEL_SAMPLES}"
    local LOG_FILE="$LOG_DIR/${MODEL_TYPE}_training.log"

    echo "Output: $OUTPUT_DIR"
    echo "Log: $LOG_FILE"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Run training
    if CUDA_VISIBLE_DEVICES=$GPU_ID python "$TRAINING_SCRIPT" \
        --mode train \
        --model-type "$MODEL_TYPE" \
        --epochs "$MODEL_EPOCHS" \
        --max-samples-per-dataset "$MODEL_SAMPLES" \
        --batch-size "$BATCH_SIZE" \
        --lora-rank "$LORA_RANK" \
        --output-dir "$OUTPUT_DIR" \
        --gpu-id 0 \
        2>&1 | tee "$LOG_FILE"; then

        local MODEL_END_TIME=$(date +%s)
        local MODEL_DURATION=$((MODEL_END_TIME - MODEL_START_TIME))
        local MODEL_DURATION_MIN=$((MODEL_DURATION / 60))
        local MODEL_DURATION_SEC=$((MODEL_DURATION % 60))

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✅ $MODEL_TYPE completed successfully!"
        echo "   Duration: ${MODEL_DURATION_MIN}m ${MODEL_DURATION_SEC}s"
        echo "   Model saved to: $OUTPUT_DIR"
        echo "   Log saved to: $LOG_FILE"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""

        return 0
    else
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "❌ $MODEL_TYPE FAILED!"
        echo "   Check log file: $LOG_FILE"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""

        return 1
    fi
}

# ============================================================================
# TRAIN ALL 6 SPECIALISTS
# ============================================================================

TOTAL_MODELS=6
COMPLETED_MODELS=0
FAILED_MODELS_COUNT=0

# Track which models succeeded/failed
declare -a SUCCESSFUL_MODELS
declare -a FAILED_MODELS

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    TRAINING ALL SPECIALISTS                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# 1. MATH REASONER
# ============================================================================

if train_specialist \
    "math-reasoner" \
    "$EPOCHS" \
    "$SAMPLES_PER_DATASET" \
    "STEM reasoning and problem-solving" \
    "ARC (AI2 Reasoning Challenge)" \
    "math, physics, engineering"; then
    SUCCESSFUL_MODELS+=("math-reasoner")
    COMPLETED_MODELS=$((COMPLETED_MODELS + 1))
else
    FAILED_MODELS+=("math-reasoner")
    FAILED_MODELS_COUNT=$((FAILED_MODELS_COUNT + 1))
fi

# ============================================================================
# 2. SCIENCE EXPERT
# ============================================================================

if train_specialist \
    "science-expert" \
    "$EPOCHS" \
    "$SAMPLES_PER_DATASET" \
    "Natural sciences and CS" \
    "ARC (1.2K) + OpenBookQA (500) + SciQ (1K)" \
    "biology, chemistry, computer science"; then
    SUCCESSFUL_MODELS+=("science-expert")
    COMPLETED_MODELS=$((COMPLETED_MODELS + 1))
else
    FAILED_MODELS+=("science-expert")
    FAILED_MODELS_COUNT=$((FAILED_MODELS_COUNT + 1))
fi

# ============================================================================
# 3. SOCIAL SCIENCES
# ============================================================================

if train_specialist \
    "social-sciences" \
    "$EPOCHS" \
    "$SAMPLES_PER_DATASET" \
    "Social sciences and human behavior" \
    "CommonsenseQA (1.2K) + StrategyQA (2.3K)" \
    "psychology, economics, business"; then
    SUCCESSFUL_MODELS+=("social-sciences")
    COMPLETED_MODELS=$((COMPLETED_MODELS + 1))
else
    FAILED_MODELS+=("social-sciences")
    FAILED_MODELS_COUNT=$((FAILED_MODELS_COUNT + 1))
fi

# ============================================================================
# 4. HUMANITIES
# ============================================================================

if train_specialist \
    "humanities" \
    "$EPOCHS" \
    "$SAMPLES_PER_DATASET" \
    "Historical and philosophical reasoning" \
    "TruthfulQA (817)" \
    "history, philosophy"; then
    SUCCESSFUL_MODELS+=("humanities")
    COMPLETED_MODELS=$((COMPLETED_MODELS + 1))
else
    FAILED_MODELS+=("humanities")
    FAILED_MODELS_COUNT=$((FAILED_MODELS_COUNT + 1))
fi

# ============================================================================
# 5. LAW
# ============================================================================

# Law uses different settings (more epochs, fewer samples)
LAW_EPOCHS=$((EPOCHS + 3))  # +3 epochs for specialized domain
LAW_SAMPLES=$((SAMPLES_PER_DATASET / 3))  # Fewer samples available

if train_specialist \
    "law" \
    "$LAW_EPOCHS" \
    "$LAW_SAMPLES" \
    "Legal reasoning and jurisprudence" \
    "MMLU validation (law only)" \
    "law"; then
    SUCCESSFUL_MODELS+=("law")
    COMPLETED_MODELS=$((COMPLETED_MODELS + 1))
else
    FAILED_MODELS+=("law")
    FAILED_MODELS_COUNT=$((FAILED_MODELS_COUNT + 1))
fi

# ============================================================================
# 6. GENERALIST
# ============================================================================

if train_specialist \
    "generalist" \
    "$EPOCHS" \
    "$SAMPLES_PER_DATASET" \
    "Mixed domains (catch-all specialist)" \
    "ARC + CommonsenseQA + TruthfulQA" \
    "health, other"; then
    SUCCESSFUL_MODELS+=("generalist")
    COMPLETED_MODELS=$((COMPLETED_MODELS + 1))
else
    FAILED_MODELS+=("generalist")
    FAILED_MODELS_COUNT=$((FAILED_MODELS_COUNT + 1))
fi

# ============================================================================
# FINAL SUMMARY
# ============================================================================

END_TIME=$(date +%s)
END_TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$((TOTAL_DURATION / 3600))
TOTAL_MINUTES=$(((TOTAL_DURATION % 3600) / 60))
TOTAL_SECONDS=$((TOTAL_DURATION % 60))

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                       TRAINING COMPLETE                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Started:  $START_TIMESTAMP"
echo "Finished: $END_TIMESTAMP"
echo "Duration: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "RESULTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Total models: $TOTAL_MODELS"
echo "Completed successfully: $COMPLETED_MODELS"
echo "Failed: $FAILED_MODELS_COUNT"
echo ""

if [ ${#SUCCESSFUL_MODELS[@]} -gt 0 ]; then
    echo "✅ Successful models:"
    for model in "${SUCCESSFUL_MODELS[@]}"; do
        echo "   - $model"
    done
    echo ""
fi

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "❌ Failed models:"
    for model in "${FAILED_MODELS[@]}"; do
        echo "   - $model"
    done
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "OUTPUT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Models saved to: $OUTPUT_BASE_DIR/"
echo "Logs saved to: $LOG_DIR/"
echo ""

# List all trained models
if [ -d "$OUTPUT_BASE_DIR" ]; then
    echo "Trained models:"
    find "$OUTPUT_BASE_DIR" -maxdepth 1 -mindepth 1 -type d -exec basename {} \; | sed 's/^/  - /'
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "NEXT STEPS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Review training logs:"
echo "   ls -lh $LOG_DIR/"
echo ""
echo "2. Check model performance:"
echo "   cat $OUTPUT_BASE_DIR/*/training_comparison.json"
echo ""
echo "3. Test individual models:"
echo "   python ft_qwen3_mmlu_solver_lora_no_leakage.py \\"
echo "       --mode test \\"
echo "       --model-path $OUTPUT_BASE_DIR/math-reasoner_r32_e5_s1000"
echo ""
echo "4. Deploy with router system (after all models trained):"
echo "   python mmlu_solver_router.py \\"
echo "       --classifier-path path/to/classifier \\"
echo "       --solver-base-path $OUTPUT_BASE_DIR/"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Create summary file
SUMMARY_FILE="$LOG_DIR/training_summary_$(date +%Y%m%d_%H%M%S).txt"
cat > "$SUMMARY_FILE" << SUMMARY_EOF
MMLU-Pro Specialists - Batch Training Summary (NO DATA LEAKAGE)
================================================================

Started:  $START_TIMESTAMP
Finished: $END_TIMESTAMP
Duration: ${TOTAL_HOURS}h ${TOTAL_MINUTES}m ${TOTAL_SECONDS}s

Configuration:
- GPU ID: $GPU_ID
- Samples per dataset: $SAMPLES_PER_DATASET
- Epochs: $EPOCHS
- Batch size: $BATCH_SIZE
- LoRA rank: $LORA_RANK

Results:
- Total models: $TOTAL_MODELS
- Completed: $COMPLETED_MODELS
- Failed: $FAILED_MODELS_COUNT

Successful models:
$(printf '%s\n' "${SUCCESSFUL_MODELS[@]}" | sed 's/^/  - /')

$(if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "Failed models:"
    printf '%s\n' "${FAILED_MODELS[@]}" | sed 's/^/  - /'
fi)

Output directory: $OUTPUT_BASE_DIR/
Log directory: $LOG_DIR/

Training Details:
1. math-reasoner: GSM8K + MATH → MMLU-Pro (math, physics, engineering)
2. science-expert: ARC + OpenBookQA + SciQ → MMLU-Pro (bio, chem, CS)
3. social-sciences: CommonsenseQA + StrategyQA → MMLU-Pro (psych, econ, biz)
4. humanities: TruthfulQA → MMLU-Pro (history, philosophy)
5. law: MMLU-train (law) → MMLU-Pro (law)
6. generalist: Mixed datasets → MMLU-Pro (health, other)

Data Leakage: ✅ NONE - Training and test datasets are completely separate!

SUMMARY_EOF

echo "Summary saved to: $SUMMARY_FILE"
echo ""

# Exit with appropriate code
if [ $FAILED_MODELS_COUNT -eq 0 ]; then
    echo "✅ All models trained successfully!"
    echo ""
    exit 0
else
    echo "⚠️  Some models failed. Check logs for details."
    echo ""
    exit 1
fi

