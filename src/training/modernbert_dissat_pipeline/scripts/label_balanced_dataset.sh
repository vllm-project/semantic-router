#!/bin/bash
#
# Label Balanced Dataset for Feedback Detector
# ============================================
# 
# This script labels examples from multiple datasets to create a balanced
# 4-class training set for the feedback detector model.
#
# Target: ~4,000 examples per class = ~16,000 total
# Classes: SAT, NEED_CLARIFICATION, WRONG_ANSWER, WANT_DIFFERENT
#
# Features:
# - Incremental persistence with checkpointing
# - Resume capability (--resume flag)
# - Garbage protection with circuit breaker
# - High throughput settings
#
# Usage:
#   ./label_balanced_dataset.sh              # Start fresh
#   ./label_balanced_dataset.sh --resume     # Resume from checkpoint
#   ./label_balanced_dataset.sh --status     # Check progress only
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"
LABELED_DIR="$DATA_DIR/labeled_data"
RAW_DIR="$DATA_DIR/raw"

# Labeling settings (high throughput)
WORKERS=8           # Increased from 4
RATE_LIMIT=20.0     # Increased from 8 requests per second
RETRIES=3
CHECKPOINT_INTERVAL=100
GARBAGE_THRESHOLD=5
GARBAGE_MAX=200
TIMEOUT=60

# Target examples per dataset for BALANCED classes
# Goal: ~4,000 per class = ~16,000 total
#
# Complaint datasets → ~70% WRONG_ANSWER
# Dialogue datasets  → ~30% SAT, ~50% NEED_CLAR, ~15% WANT_DIFF
#
declare -A DATASET_TARGETS=(
    # Complaint datasets (REDUCED for balance)
    # 5,000 total → ~3,500 WRONG_ANSWER
    ["consumer_complaints_medium"]=2000
    ["customer_complaints"]=2000
    ["turkish_complaints"]=1000
    
    # Dialogue datasets (INCREASED for SAT/WANT_DIFF)
    # 13,000 total → ~3,900 SAT, ~6,500 NEED_CLAR, ~2,000 WANT_DIFF
    ["multiwoz"]=3000
    ["sgd"]=3000
    ["inscit"]=2500
    ["mimics"]=2000
    ["hazumi"]=1500
    ["redial"]=1000
)

# API settings
API_URL="${VLLM_API_URL:-http://localhost:8000/v1/chat/completions}"
MODEL="${VLLM_MODEL:-openai/gpt-oss-120b}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Function to check model health
check_model_health() {
    log_info "Checking model health..."
    
    response=$(curl -s --max-time 30 "$API_URL" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say OK\"}],
            \"temperature\": 0,
            \"max_tokens\": 10
        }" 2>/dev/null || echo "")
    
    if [ -z "$response" ]; then
        log_error "Model not responding at $API_URL"
        return 1
    fi
    
    content=$(echo "$response" | python3 -c "
import sys, json
try:
    r = json.load(sys.stdin)
    c = r['choices'][0]['message']['content']
    if len(set(c)) < 3:
        print('GARBAGE')
    else:
        print('OK')
except:
    print('ERROR')
" 2>/dev/null)
    
    if [ "$content" = "OK" ]; then
        log_success "Model is healthy"
        return 0
    elif [ "$content" = "GARBAGE" ]; then
        log_error "Model returning garbage output"
        return 1
    else
        log_error "Model health check failed"
        return 1
    fi
}

# Function to show current progress
show_progress() {
    echo ""
    echo "========================================"
    echo "  LABELING PROGRESS"
    echo "========================================"
    echo ""
    
    total_labeled=0
    total_sat=0
    total_need=0
    total_wrong=0
    total_want=0
    total_errors=0
    
    for dataset in "${!DATASET_TARGETS[@]}"; do
        checkpoint="$LABELED_DIR/${dataset}_checkpoint.json"
        if [ -f "$checkpoint" ]; then
            stats=$(python3 -c "
import json
with open('$checkpoint') as f:
    d = json.load(f)
    s = d.get('stats', {})
    t = s.get('total', 0)
    sat = s.get('SAT', 0)
    need = s.get('NEED_CLARIFICATION', 0)
    wrong = s.get('WRONG_ANSWER', 0)
    want = s.get('WANT_DIFFERENT', 0)
    err = s.get('errors', 0)
    print(f'{t}|{sat}|{need}|{wrong}|{want}|{err}')
" 2>/dev/null)
            
            IFS='|' read -r t sat need wrong want err <<< "$stats"
            printf "  %-30s %5d labeled (SAT:%d NEED:%d WRONG:%d WANT:%d err:%d)\n" \
                "$dataset:" "$t" "$sat" "$need" "$wrong" "$want" "$err"
            
            total_labeled=$((total_labeled + t))
            total_sat=$((total_sat + sat))
            total_need=$((total_need + need))
            total_wrong=$((total_wrong + wrong))
            total_want=$((total_want + want))
            total_errors=$((total_errors + err))
        else
            printf "  %-30s %5s\n" "$dataset:" "not started"
        fi
    done
    
    echo ""
    echo "----------------------------------------"
    printf "  %-30s %5d\n" "TOTAL LABELED:" "$total_labeled"
    echo ""
    echo "  Label Distribution:"
    printf "    SAT:               %5d\n" "$total_sat"
    printf "    NEED_CLARIFICATION:%5d\n" "$total_need"
    printf "    WRONG_ANSWER:      %5d\n" "$total_wrong"
    printf "    WANT_DIFFERENT:    %5d\n" "$total_want"
    printf "    Errors:            %5d\n" "$total_errors"
    echo "========================================"
    echo ""
}

# Function to label a single dataset
label_dataset() {
    local dataset=$1
    local max_examples=$2
    local resume_flag=$3
    
    log_info "Labeling $dataset (max: $max_examples)..."
    
    cd "$PROJECT_DIR"
    
    # Build resume argument
    resume_arg=""
    if [ "$resume_flag" = "true" ]; then
        resume_arg="--resume"
    else
        resume_arg="--no-resume"
    fi
    
    # Run labeling for specific dataset
    python3 -c "
import sys
sys.path.append('.')
from data_processing.download_datasets import (
    label_dataset,
    LabelingConfig,
    DataConfig,
    extract_mimics_examples,
    extract_inscit_examples,
    extract_multiwoz_examples,
    extract_sgd_examples,
    extract_redial_examples,
    extract_hazumi_examples,
    extract_customer_complaints_examples,
    extract_consumer_complaints_medium_examples,
    extract_turkish_complaints_examples,
)
import os

# Extractor mapping
EXTRACTORS = {
    'mimics': extract_mimics_examples,
    'inscit': extract_inscit_examples,
    'multiwoz': extract_multiwoz_examples,
    'sgd': extract_sgd_examples,
    'redial': extract_redial_examples,
    'hazumi': extract_hazumi_examples,
    'customer_complaints': extract_customer_complaints_examples,
    'consumer_complaints_medium': extract_consumer_complaints_medium_examples,
    'turkish_complaints': extract_turkish_complaints_examples,
}

dataset = '$dataset'
max_examples = $max_examples
resume = '$resume_flag' == 'true'

config = DataConfig()
extractor = EXTRACTORS.get(dataset)

if not extractor:
    print(f'Unknown dataset: {dataset}')
    sys.exit(1)

# Check if dataset exists
dataset_dir = os.path.join(config.raw_data_dir, dataset)
if not os.path.exists(dataset_dir):
    print(f'Dataset not downloaded: {dataset}')
    sys.exit(1)

# Extract examples
print(f'Extracting up to {max_examples} examples from {dataset}...')
examples = []
for ex in extractor(config.raw_data_dir):
    examples.append(ex)
    if len(examples) >= max_examples:
        break

if not examples:
    print(f'No examples found in {dataset}')
    sys.exit(1)

print(f'Extracted {len(examples)} examples')

# Labeling config
label_config = LabelingConfig(
    api_url='$API_URL',
    model='$MODEL',
    workers=$WORKERS,
    rate_limit_rps=$RATE_LIMIT,
    max_retries=$RETRIES,
    checkpoint_interval=$CHECKPOINT_INTERVAL,
    garbage_consecutive_threshold=$GARBAGE_THRESHOLD,
    garbage_max_total=$GARBAGE_MAX,
    timeout=$TIMEOUT,
)

# Output path
output_dir = '$LABELED_DIR'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'{dataset}_labeled.jsonl')

# Clear if not resuming
if not resume:
    checkpoint_file = os.path.join(output_dir, f'{dataset}_checkpoint.json')
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    if os.path.exists(output_path):
        os.remove(output_path)

# Label
print(f'Labeling {dataset}...')
stats = label_dataset(examples, output_path, label_config, dataset)

print(f'\\nResults for {dataset}:')
print(f'  Total: {stats[\"total\"]}')
print(f'  SAT: {stats[\"SAT\"]}')
print(f'  NEED_CLARIFICATION: {stats[\"NEED_CLARIFICATION\"]}')
print(f'  WRONG_ANSWER: {stats[\"WRONG_ANSWER\"]}')
print(f'  WANT_DIFFERENT: {stats[\"WANT_DIFFERENT\"]}')
print(f'  Errors: {stats[\"errors\"]}')
"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        log_success "Completed $dataset"
    else
        log_error "Failed $dataset (exit code: $exit_code)"
    fi
    
    return $exit_code
}

# Function to combine all labeled data
combine_datasets() {
    log_info "Combining all labeled datasets..."
    
    combined_file="$LABELED_DIR/combined_labeled.jsonl"
    : > "$combined_file"  # Clear file
    
    total_lines=0
    for dataset in "${!DATASET_TARGETS[@]}"; do
        labeled_file="$LABELED_DIR/${dataset}_labeled.jsonl"
        if [ -f "$labeled_file" ]; then
            lines=$(wc -l < "$labeled_file")
            cat "$labeled_file" >> "$combined_file"
            total_lines=$((total_lines + lines))
            log_info "  Added $lines examples from $dataset"
        fi
    done
    
    log_success "Combined $total_lines examples to $combined_file"
    
    # Show final distribution
    python3 -c "
import json
from collections import Counter

labels = Counter()
with open('$combined_file') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            label = data.get('label_name', 'unknown')
            labels[label] += 1

print('\\nFinal Label Distribution:')
for label, count in sorted(labels.items()):
    print(f'  {label}: {count}')
print(f'  TOTAL: {sum(labels.values())}')
"
}

# Main execution
main() {
    local resume_mode=false
    local status_only=false
    
    # Parse arguments
    for arg in "$@"; do
        case $arg in
            --resume)
                resume_mode=true
                ;;
            --status)
                status_only=true
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --resume    Resume from checkpoint"
                echo "  --status    Show progress only"
                echo "  --help      Show this help"
                exit 0
                ;;
        esac
    done
    
    # Show status if requested
    if [ "$status_only" = true ]; then
        show_progress
        exit 0
    fi
    
    echo ""
    echo "========================================"
    echo "  BALANCED DATASET LABELING"
    echo "========================================"
    echo ""
    echo "  API URL: $API_URL"
    echo "  Model:   $MODEL"
    echo "  Workers: $WORKERS"
    echo "  Rate:    $RATE_LIMIT req/s"
    echo "  Resume:  $resume_mode"
    echo ""
    
    # Create directories
    mkdir -p "$LABELED_DIR"
    
    # Check model health
    if ! check_model_health; then
        log_error "Please start the vLLM server and try again"
        exit 1
    fi
    
    # Label each dataset
    log_info "Starting labeling process..."
    echo ""
    
    # Process datasets in order (complaints first for WRONG_ANSWER)
    datasets_order=(
        "consumer_complaints_medium"
        "customer_complaints"
        "turkish_complaints"
        "multiwoz"
        "sgd"
        "inscit"
        "mimics"
        "hazumi"
        "redial"
    )
    
    failed_datasets=()
    
    for dataset in "${datasets_order[@]}"; do
        max_examples=${DATASET_TARGETS[$dataset]:-1000}
        
        echo ""
        echo "----------------------------------------"
        
        if ! label_dataset "$dataset" "$max_examples" "$resume_mode"; then
            failed_datasets+=("$dataset")
            log_warn "Continuing with next dataset..."
        fi
        
        # Show progress after each dataset
        show_progress
    done
    
    # Combine results
    echo ""
    combine_datasets
    
    # Final summary
    echo ""
    echo "========================================"
    echo "  LABELING COMPLETE"
    echo "========================================"
    
    if [ ${#failed_datasets[@]} -gt 0 ]; then
        log_warn "Some datasets failed: ${failed_datasets[*]}"
        log_info "Run with --resume to retry failed datasets"
    else
        log_success "All datasets labeled successfully!"
    fi
    
    show_progress
}

# Run main
main "$@"
