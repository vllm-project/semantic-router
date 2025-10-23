#!/bin/bash

# CPU-Optimized Training Script for Security Detection LoRA
# ==========================================================
#
# This script is optimized for training on CPU without GPU memory.
# It uses smaller models, reduced batch sizes, and CPU-friendly parameters.

set -e

echo "🖥️  CPU-Optimized Security Detection LoRA Training"
echo "================================================="

# CPU-optimized configuration
EPOCHS=8                     # Reduced epochs for faster training
LORA_RANK=8                  # Optimal rank for stability and performance
LORA_ALPHA=16                # Standard alpha (2x rank) for best results
MAX_SAMPLES=7000             # Increased samples for better security detection coverage
BATCH_SIZE=2                 # Small batch size for CPU
LEARNING_RATE=3e-5           # Optimized learning rate based on  PEFT best practices


CPU_MODELS=(
    "bert-base-uncased"     # 110M params - most CPU-friendly, needs retraining with fixed config
    "roberta-base"          # 125M params - better security detection performance, proven stable
    "modernbert-base"
)

# Parse command line arguments
MODELS=("${CPU_MODELS[@]}")
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            shift
            MODELS=()
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --rank)
            LORA_RANK="$2"
            LORA_ALPHA=$((LORA_RANK * 2))  # Auto-adjust alpha
            shift 2
            ;;
        --quick)
            EPOCHS=3
            MAX_SAMPLES=500
            BATCH_SIZE=1
            echo "⚡ Ultra-quick CPU mode: $EPOCHS epochs, $MAX_SAMPLES samples"
            ;;
        --help)
            echo "CPU-Optimized Security Detection LoRA Training"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --models MODEL1 MODEL2    Specify models to train"
            echo "  --epochs N                Number of epochs (default: $EPOCHS)"
            echo "  --samples N               Max samples (default: $MAX_SAMPLES)"
            echo "  --batch-size N            Batch size (default: $BATCH_SIZE)"
            echo "  --rank N                  LoRA rank (default: $LORA_RANK)"
            echo "  --quick                   Ultra-quick mode for testing"
            echo "  --help                    Show this help"
            echo ""
            echo "CPU-friendly models: bert-base-uncased, roberta-base"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🔧 CPU Training Configuration:"
echo "   Models: ${MODELS[*]}"
echo "   Epochs: $EPOCHS"
echo "   LoRA Rank: $LORA_RANK (Alpha: $LORA_ALPHA)"
echo "   Max Samples: $MAX_SAMPLES"
echo "   Batch Size: $BATCH_SIZE"
echo "   Learning Rate: $LEARNING_RATE"
echo "   🖥️  Device: CPU (no GPU required)"
echo ""

# Estimate training time
model_count=${#MODELS[@]}
estimated_minutes=$((model_count * EPOCHS * MAX_SAMPLES / 100))
echo "⏱️  Estimated training time: ~${estimated_minutes} minutes"
echo ""

# Create results directory
RESULTS_DIR="cpu_training_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
echo "📁 Results will be saved to: $RESULTS_DIR"

# Initialize summary file
SUMMARY_FILE="$RESULTS_DIR/cpu_training_summary.txt"
{
    echo "Security Detection LoRA - CPU Training Summary"
    echo "==============================================="
    echo "Date: $(date)"
    echo "Models: ${MODELS[*]}"
    echo "CPU-optimized parameters: epochs=$EPOCHS, rank=$LORA_RANK, samples=$MAX_SAMPLES, batch=$BATCH_SIZE"
    echo ""
} > "$SUMMARY_FILE"

# Function to train a single model on CPU
train_cpu_model() {
    local model_name=$1
    local start_time=$(date +%s)

    echo ""
    echo "🚀 Training model on CPU: $model_name"
    echo "⏰ Start time: $(date)"

    # Create model-specific log file
    local log_file="$RESULTS_DIR/${model_name}_cpu_training.log"

    # CPU-optimized training command
    local cmd="python jailbreak_bert_finetuning_lora.py \
        --mode train \
        --model $model_name \
        --epochs $EPOCHS \
        --lora-rank $LORA_RANK \
        --lora-alpha $LORA_ALPHA \
        --max-samples $MAX_SAMPLES \
        --batch-size $BATCH_SIZE \
        --learning-rate $LEARNING_RATE"

    echo "📝 Command: $cmd"
    echo "📋 Log file: $log_file"
    echo "🖥️  Training on CPU (this may take longer than GPU)..."

    # Set environment variables to force CPU usage
    export CUDA_VISIBLE_DEVICES=""
    export OMP_NUM_THREADS=4  # Optimize CPU threads

    # Run training and capture result
    if eval "$cmd" > "$log_file" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local minutes=$((duration / 60))
        local seconds=$((duration % 60))

        echo "✅ SUCCESS: $model_name trained on CPU in ${minutes}m ${seconds}s"
        echo "$model_name: SUCCESS (${minutes}m ${seconds}s)" >> "$SUMMARY_FILE"

        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local minutes=$((duration / 60))
        local seconds=$((duration % 60))

        echo "❌ FAILED: $model_name failed after ${minutes}m ${seconds}s"
        echo "$model_name: FAILED (${minutes}m ${seconds}s)" >> "$SUMMARY_FILE"

        # Show last few lines of error log
        echo "🔍 Last 10 lines of error log:"
        tail -10 "$log_file"

        return 1
    fi
}

# Function to test a trained model
test_cpu_model() {
    local model_name=$1
    local python_model_dir="lora_jailbreak_classifier_${model_name}_r${LORA_RANK}_model"
    local rust_model_dir="lora_jailbreak_classifier_${model_name}_r${LORA_RANK}_model_rust"

    echo ""
    echo "🔍 Testing model on CPU: $model_name"

    # Test Python model first
    if [[ -d "$python_model_dir" ]]; then
        echo "  📝 Testing Python inference..."
        local python_test_log="$RESULTS_DIR/${model_name}_python_test.log"

        # Force CPU for testing
        export CUDA_VISIBLE_DEVICES=""
        local python_cmd="python jailbreak_bert_finetuning_lora.py --mode test --model-path $python_model_dir"

        if eval "$python_cmd" > "$python_test_log" 2>&1; then
            echo "  ✅ Python test completed"

            # Extract key metrics
            local predictions_count=$(grep -c "Prediction:" "$python_test_log" 2>/dev/null || echo "0")
            local low_confidence=$(grep -c "confidence: 0\.[0-4]" "$python_test_log" 2>/dev/null || echo "0")

            echo "  📊 Python Results: $predictions_count predictions made, $low_confidence low confidence predictions"
            echo "$model_name: Python Test OK ($predictions_count predictions, $low_confidence low conf)" >> "$SUMMARY_FILE"
        else
            echo "  ❌ Python test failed"
            echo "$model_name: Python Test FAILED" >> "$SUMMARY_FILE"
        fi
    else
        echo "  ⚠️  Python model directory not found: $python_model_dir"
    fi

    # Test Go model if available
    if [[ -d "$rust_model_dir" ]]; then
        echo "  🦀 Testing Go inference..."
        local go_test_log="$RESULTS_DIR/${model_name}_go_test.log"

        # Force CPU for testing
        export CUDA_VISIBLE_DEVICES=""
        export LD_LIBRARY_PATH="../../../../candle-binding/target/release"
        local go_cmd="go run jailbreak_bert_finetuning_lora_verifier.go -jailbreak-model $rust_model_dir"

        if eval "$go_cmd" > "$go_test_log" 2>&1; then
            echo "  ✅ Go test completed"
            echo "$model_name: Go Test OK" >> "$SUMMARY_FILE"
        else
            echo "  ❌ Go test failed"
            echo "$model_name: Go Test FAILED" >> "$SUMMARY_FILE"
        fi
    else
        echo "  ⚠️  Go model directory not found: $rust_model_dir"
    fi
}

# Main training loop
echo "🎯 Starting CPU training for ${#MODELS[@]} models..."
echo "⚠️  Note: CPU training is slower than GPU but uses no GPU memory"
echo ""

successful_models=()
failed_models=()

for model in "${MODELS[@]}"; do
    if train_cpu_model "$model"; then
        successful_models+=("$model")
    else
        failed_models+=("$model")
    fi

    # Small delay between trainings
    sleep 2
done

# Summary
echo ""
echo "📊 CPU TRAINING SUMMARY:"
echo "======================="
echo "✅ Successful: ${#successful_models[@]} models"
echo "❌ Failed: ${#failed_models[@]} models"

if [[ ${#successful_models[@]} -gt 0 ]]; then
    echo ""
    echo "✅ Successful models:"
    for model in "${successful_models[@]}"; do
        echo "   • $model"
    done
fi

if [[ ${#failed_models[@]} -gt 0 ]]; then
    echo ""
    echo "❌ Failed models:"
    for model in "${failed_models[@]}"; do
        echo "   • $model"
    done
fi

# Test successful models
if [[ ${#successful_models[@]} -gt 0 ]]; then
    echo ""
    echo "🔍 Testing successful models on CPU..."
    {
        echo ""
        echo "CPU Testing Results:"
        echo "==================="
    } >> "$SUMMARY_FILE"

    for model in "${successful_models[@]}"; do
        test_cpu_model "$model"
    done
fi

# Final summary
echo ""
echo "🎉 CPU training completed!"
echo "📁 Results saved in: $RESULTS_DIR"
echo "📋 Summary file: $SUMMARY_FILE"
echo ""
echo "💡 CPU Training Tips:"
echo "   • CPU training is slower but uses no GPU memory"
echo "   • Consider using --quick mode for initial testing"
echo "   • bert-base-uncased is usually the most CPU-friendly and stable"
echo "   • roberta-base may have better security detection accuracy"
echo "   • You can increase --batch-size if you have more RAM"
echo ""

# Display final summary
echo "📊 FINAL CPU TRAINING SUMMARY:"
cat "$SUMMARY_FILE"
