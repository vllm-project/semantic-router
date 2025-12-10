#!/bin/bash
# Setup script to download datasets that require git cloning
# Run this ONCE before training to cache datasets locally
#
# Usage: ./setup_datasets.sh [--data-dir /path/to/cache]
#
# This script downloads:
# 1. NISQ Dataset (ISQ vs Non-ISQ classification - gold standard)
# 2. FaithDial (Information-seeking dialogue benchmark)
# 3. FactCHD (Fact-conflicting hallucination detection)

set -e

# Default data directory
DATA_DIR="${1:-./datasets_cache}"

echo "=============================================="
echo "Fact-Check Classifier Dataset Setup"
echo "=============================================="
echo "Data directory: $DATA_DIR"
echo ""

# Create data directory
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Function to clone or update a repo
clone_or_update() {
    local repo_url="$1"
    local repo_name="$2"
    local branch="${3:-main}"
    
    if [ -d "$repo_name" ]; then
        echo "[$repo_name] Already exists, updating..."
        cd "$repo_name"
        git pull origin "$branch" 2>/dev/null || echo "  (pull failed, using existing)"
        cd ..
    else
        echo "[$repo_name] Cloning from $repo_url..."
        git clone --depth 1 "$repo_url" "$repo_name"
    fi
}

echo ""
echo "=== Downloading NISQ Dataset ==="
echo "Reference: 'What Are the Implications of Your Question?'"
echo "           Non-Information Seeking Question-Type Identification (ACL LREC 2024)"
clone_or_update "https://github.com/YaoSun0422/NISQ_dataset.git" "NISQ_dataset"
if [ -f "NISQ_dataset/final_train.csv" ]; then
    echo "  ✓ NISQ dataset ready (final_train.csv found)"
    wc -l NISQ_dataset/final_train.csv | awk '{print "    Lines: " $1}'
else
    echo "  ✗ Warning: final_train.csv not found"
fi

echo ""
echo "=== Downloading FaithDial Dataset ==="
echo "Reference: 'FaithDial: A Benchmark for Information-Seeking Dialogue' (TACL 2022)"
clone_or_update "https://huggingface.co/datasets/McGill-NLP/FaithDial" "FaithDial_dataset"
if [ -f "FaithDial_dataset/train.json" ]; then
    echo "  ✓ FaithDial dataset ready (train.json found)"
else
    echo "  ✗ Warning: train.json not found"
fi

echo ""
echo "=== Downloading FactCHD Dataset ==="
echo "Reference: 'FactCHD: Benchmarking Fact-Conflicting Hallucination Detection' (2024)"
clone_or_update "https://huggingface.co/datasets/zjunlp/FactCHD" "FactCHD_dataset"
if [ -f "FactCHD_dataset/fact_train_noe.jsonl" ]; then
    echo "  ✓ FactCHD dataset ready (fact_train_noe.jsonl found)"
    wc -l FactCHD_dataset/fact_train_noe.jsonl | awk '{print "    Lines: " $1}'
else
    echo "  ✗ Warning: fact_train_noe.jsonl not found"
fi

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Datasets cached in: $DATA_DIR"
echo ""
echo "To use with training, set the environment variable:"
echo "  export FACT_CHECK_DATA_DIR=$DATA_DIR"
echo ""
echo "Or run training with:"
echo "  python fact_check_bert_finetuning_lora.py --mode train --data-dir $DATA_DIR"
echo ""

# Show disk usage
echo "Disk usage:"
du -sh "$DATA_DIR"/* 2>/dev/null || echo "  (no data yet)"

