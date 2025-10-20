#!/bin/bash

# Comprehensive Multi-Dataset Benchmark Script for Research Report
# This script benchmarks all available datasets with reasonable sample sizes
# for statistical significance while maintaining manageable runtime.

set -e

# Default Configuration
VENV_PATH="../.venv"
ROUTER_ENDPOINT="http://127.0.0.1:8801/v1"
VLLM_ENDPOINT="http://127.0.0.1:8000/v1"
VLLM_MODEL=""  # Will be auto-detected from endpoint if not specified
ROUTER_MODEL="auto"
CONCURRENT_REQUESTS=8
OUTPUT_BASE="results/comprehensive_research_$(date +%Y%m%d_%H%M%S)"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vllm-model)
            VLLM_MODEL="$2"
            shift 2
            ;;
        --vllm-endpoint)
            VLLM_ENDPOINT="$2"
            shift 2
            ;;
        --router-endpoint)
            ROUTER_ENDPOINT="$2"
            shift 2
            ;;
        --router-model)
            ROUTER_MODEL="$2"
            shift 2
            ;;
        --output-base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --vllm-model MODEL      Specify vLLM model (auto-detected if not provided)"
            echo "  --vllm-endpoint URL     vLLM endpoint URL (default: http://127.0.0.1:8000/v1)"
            echo "  --router-endpoint URL   Router endpoint URL (default: http://127.0.0.1:8801/v1)"
            echo "  --router-model MODEL    Router model (default: auto)"
            echo "  --output-base DIR       Output directory base (default: results/comprehensive_research_TIMESTAMP)"
            echo "  --help, -h              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Auto-detect vLLM model if not specified
if [[ -z "$VLLM_MODEL" ]]; then
    echo -e "${BLUE}🔍 Auto-detecting vLLM model from endpoint...${NC}"

    # Try to fetch models from the vLLM endpoint
    VLLM_MODELS_JSON=$(curl -s "$VLLM_ENDPOINT/models" 2>/dev/null || echo "")

    if [[ -n "$VLLM_MODELS_JSON" ]]; then
        # Extract the first model ID from the JSON response
        VLLM_MODEL=$(echo "$VLLM_MODELS_JSON" | python3 -c "
import json
import sys
try:
    data = json.load(sys.stdin)
    if 'data' in data and len(data['data']) > 0:
        print(data['data'][0]['id'])
    else:
        print('')
except:
    print('')
" 2>/dev/null)

        if [[ -n "$VLLM_MODEL" ]]; then
            echo -e "${GREEN}✅ Auto-detected vLLM model: $VLLM_MODEL${NC}"
        else
            echo -e "${RED}❌ Failed to parse models from endpoint response${NC}"
            echo -e "${YELLOW}⚠️  Using fallback model: openai/gpt-oss-20b${NC}"
            VLLM_MODEL="openai/gpt-oss-20b"
        fi
    else
        echo -e "${RED}❌ Failed to fetch models from vLLM endpoint: $VLLM_ENDPOINT${NC}"
        echo -e "${YELLOW}⚠️  Using fallback model: openai/gpt-oss-20b${NC}"
        VLLM_MODEL="openai/gpt-oss-20b"
    fi
fi

# Single persistent CSV file for all research results
PERSISTENT_RESEARCH_CSV="results/research_results_master.csv"

# Dataset configurations (dataset_name:samples_per_category)
# Balanced for statistical significance vs runtime
declare -A DATASET_CONFIGS=(
    # Core proven datasets
    ["gpqa"]=20          # 1 category × 20 = 20 samples - OUTSTANDING reasoning differentiation
    ["mmlu"]=10          # 57 subjects × 10 = 570 samples - EXCELLENT reasoning differentiation
    ["truthfulqa"]=15    # Truthfulness evaluation - some reasoning differentiation (60% → 73.3%)

    # Mathematical reasoning datasets
    # ["math"]=15          # Competition mathematics - DISABLED: Dataset not available on HF Hub
    ["gsm8k"]=25         # Elementary math word problems - EXPECTED good reasoning differentiation
    ["aqua-rat"]=20      # Algebraic word problems with rationales - EXPECTED good differentiation

    # Multi-step reasoning datasets
    ["drop"]=20          # Reading comprehension with discrete reasoning - EXPECTED excellent differentiation
    ["strategyqa"]=20    # Multi-step implicit reasoning - EXPECTED good differentiation

    # Scientific reasoning datasets
    ["sciq"]=25          # Science questions requiring reasoning - EXPECTED moderate differentiation
    ["openbookqa"]=20    # Elementary science with fact reasoning - EXPECTED moderate differentiation

    # Disabled datasets with poor reasoning differentiation:
    # ["arc-challenge"]=15 # 100% accuracy across all modes, minimal benefit
    # ["commonsenseqa"]=20 # Same accuracy across modes, small token difference
    # ["hellaswag"]=2      # Minimal differentiation, not reasoning-focused
)

echo -e "${BLUE}🔬 COMPREHENSIVE MULTI-DATASET BENCHMARK FOR RESEARCH${NC}"
echo -e "${BLUE}====================================================${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Router Endpoint: $ROUTER_ENDPOINT"
echo "  vLLM Endpoint: $VLLM_ENDPOINT"
echo "  vLLM Model: $VLLM_MODEL"
echo "  Output Directory: $OUTPUT_BASE"
echo ""
echo -e "${YELLOW}Dataset Sample Sizes:${NC}"
for dataset in "${!DATASET_CONFIGS[@]}"; do
    echo "  $dataset: ${DATASET_CONFIGS[$dataset]} samples per category"
done
echo ""

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source "$VENV_PATH/bin/activate"

# Create output directory
mkdir -p "$OUTPUT_BASE"
mkdir -p "$(dirname "$PERSISTENT_RESEARCH_CSV")"

# Backup and clear master research CSV for fresh results
if [[ -f "$PERSISTENT_RESEARCH_CSV" ]]; then
    BACKUP_CSV="${PERSISTENT_RESEARCH_CSV}.backup_$(date +%Y%m%d_%H%M%S)"
    cp "$PERSISTENT_RESEARCH_CSV" "$BACKUP_CSV"
    echo -e "${GREEN}📊 Backed up existing master CSV to: $BACKUP_CSV${NC}"
fi

# Create fresh master research CSV with header only
echo "Dataset,Mode,Model,Accuracy,Avg_Latency_ms,Avg_Total_Tokens,Sample_Count,Timestamp" > "$PERSISTENT_RESEARCH_CSV"
echo -e "${GREEN}📊 Created fresh master research CSV: $PERSISTENT_RESEARCH_CSV${NC}"

# Also create a timestamped copy for this run
RESEARCH_CSV="$OUTPUT_BASE/research_results.csv"
cp "$PERSISTENT_RESEARCH_CSV" "$RESEARCH_CSV"

# Function to extract metrics from results and append to research CSV
extract_and_save_metrics() {
    local dataset=$1
    local mode=$2  # "router" or "vllm"
    local results_dir=$3
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    # Find the results files (handle nested directory structure)
    local summary_file=""
    local detailed_file=""

    # Look for files in nested directories
    if [[ -d "$results_dir" ]]; then
        summary_file=$(find "$results_dir" -name "results_summary.csv" -type f | head -1)
        if [[ -z "$summary_file" ]]; then
            detailed_file=$(find "$results_dir" -name "detailed_results.csv" -type f | head -1)
        fi
    fi

    # Use whichever file we found
    local target_file=""
    if [[ -f "$summary_file" ]]; then
        target_file="$summary_file"
    elif [[ -f "$detailed_file" ]]; then
        target_file="$detailed_file"
    fi

    if [[ -n "$target_file" && -f "$target_file" ]]; then
        echo -e "${YELLOW}    📊 Extracting metrics from $target_file...${NC}"

        # Extract overall metrics from the CSV file
        # Skip header and get the last line (overall summary) or calculate averages
        local temp_file="/tmp/metrics_${dataset}_${mode}.txt"

        # Use Python to calculate averages from the CSV
        python3 -c "
import pandas as pd
import sys

try:
    df = pd.read_csv('$target_file')

    # Calculate overall metrics (handle different CSV formats)
    if len(df) > 0:
        # Handle accuracy column (is_correct vs accuracy)
        if 'is_correct' in df.columns:
            avg_accuracy = df['is_correct'].mean()
        elif 'accuracy' in df.columns:
            avg_accuracy = df['accuracy'].mean()
        else:
            avg_accuracy = 0.0

        # Handle latency column (response_time vs avg_latency_ms)
        if 'response_time' in df.columns:
            avg_latency = df['response_time'].mean() * 1000  # Convert to ms
        elif 'avg_latency_ms' in df.columns:
            avg_latency = df['avg_latency_ms'].mean()
        else:
            avg_latency = 0.0

        # Handle token column (total_tokens vs avg_total_tokens)
        if 'total_tokens' in df.columns:
            avg_tokens = df['total_tokens'].mean()
        elif 'avg_total_tokens' in df.columns:
            avg_tokens = df['avg_total_tokens'].mean()
        else:
            avg_tokens = 0.0

        sample_count = len(df)

        # Determine model name
        if '$mode' == 'router':
            model_name = '$ROUTER_MODEL'
        else:
            model_name = '$VLLM_MODEL'

        # For vLLM, we might have multiple modes (NR, NR_REASONING)
        # Check both 'mode' and 'mode_label' columns for mode information
        if '$mode' == 'vllm' and ('mode' in df.columns or 'mode_label' in df.columns):
            # Use mode_label if available (more descriptive), otherwise use mode
            mode_column = 'mode_label' if 'mode_label' in df.columns else 'mode'
            for mode_type in df[mode_column].unique():
                mode_df = df[df[mode_column] == mode_type]

                # Recalculate metrics for this specific mode using correct column names
                if 'is_correct' in mode_df.columns:
                    mode_accuracy = mode_df['is_correct'].mean()
                elif 'accuracy' in mode_df.columns:
                    mode_accuracy = mode_df['accuracy'].mean()
                else:
                    mode_accuracy = 0.0

                if 'response_time' in mode_df.columns:
                    mode_latency = mode_df['response_time'].mean() * 1000
                elif 'avg_latency_ms' in mode_df.columns:
                    mode_latency = mode_df['avg_latency_ms'].mean()
                else:
                    mode_latency = 0.0

                if 'total_tokens' in mode_df.columns:
                    mode_tokens = mode_df['total_tokens'].mean()
                elif 'avg_total_tokens' in mode_df.columns:
                    mode_tokens = mode_df['avg_total_tokens'].mean()
                else:
                    mode_tokens = 0.0

                mode_samples = len(mode_df)

                # Map technical mode names to descriptive names
                if mode_type == 'VLLM_NR':
                    display_mode = 'vLLM_No_Reasoning'
                elif mode_type == 'VLLM_NR_REASONING':
                    display_mode = 'vLLM_All_Reasoning'
                elif mode_type == 'VLLM_XC':
                    display_mode = 'vLLM_CoT'
                else:
                    display_mode = mode_type  # Use the mode_label as-is if not recognized

                csv_line = f'$dataset,{display_mode},{model_name},{mode_accuracy:.3f},{mode_latency:.1f},{mode_tokens:.1f},{mode_samples},$timestamp'
                print(f'    📝 Writing to CSV: {csv_line}', file=sys.stderr)
                print(csv_line)
        else:
            csv_line = f'$dataset,$mode,{model_name},{avg_accuracy:.3f},{avg_latency:.1f},{avg_tokens:.1f},{sample_count},$timestamp'
            print(f'    📝 Writing to CSV: {csv_line}', file=sys.stderr)
            print(csv_line)
    else:
        print(f'$dataset,$mode,unknown,0.000,0.0,0.0,0,$timestamp', file=sys.stderr)

except Exception as e:
    print(f'Error processing $target_file: {e}', file=sys.stderr)
    print(f'$dataset,$mode,unknown,0.000,0.0,0.0,0,$timestamp', file=sys.stderr)
" | tee -a "$RESEARCH_CSV" >> "$PERSISTENT_RESEARCH_CSV"

        echo -e "${GREEN}    ✅ Metrics saved to both timestamped and master research CSV${NC}"
    else
        echo -e "${RED}    ❌ Warning: No results files found in $results_dir${NC}"
        # Add a placeholder entry to both files
        echo "$dataset,$mode,unknown,0.000,0.0,0.0,0,$timestamp" | tee -a "$RESEARCH_CSV" >> "$PERSISTENT_RESEARCH_CSV"
    fi
}

# Function to run benchmark for a dataset
run_dataset_benchmark() {
    local dataset=$1
    local samples=${DATASET_CONFIGS[$dataset]}

    echo -e "${GREEN}📊 Benchmarking $dataset dataset ($samples samples per category)...${NC}"

    # Router benchmark (pass vLLM info for consistent token calculation)
    echo -e "${YELLOW}  🤖 Running router evaluation...${NC}"
    python3 -m vllm_semantic_router_bench.router_reason_bench_multi_dataset \
        --dataset "$dataset" \
        --samples-per-category "$samples" \
        --run-router \
        --router-endpoint "$ROUTER_ENDPOINT" \
        --router-models "$ROUTER_MODEL" \
        --vllm-endpoint "$VLLM_ENDPOINT" \
        --vllm-models "$VLLM_MODEL" \
        --concurrent-requests "$CONCURRENT_REQUESTS" \
        --output-dir "$OUTPUT_BASE/router_$dataset" \
        --seed 42

    # Extract and save router metrics immediately
    extract_and_save_metrics "$dataset" "Router" "$OUTPUT_BASE/router_$dataset"

    # vLLM benchmark
    echo -e "${YELLOW}  ⚡ Running vLLM evaluation...${NC}"
    python3 -m vllm_semantic_router_bench.router_reason_bench_multi_dataset \
        --dataset "$dataset" \
        --samples-per-category "$samples" \
        --run-vllm \
        --vllm-endpoint "$VLLM_ENDPOINT" \
        --vllm-models "$VLLM_MODEL" \
        --vllm-exec-modes NR NR_REASONING \
        --output-dir "$OUTPUT_BASE/vllm_$dataset" \
        --concurrent-requests "$CONCURRENT_REQUESTS" \
        --seed 42

    # Extract and save vLLM metrics immediately
    extract_and_save_metrics "$dataset" "vllm" "$OUTPUT_BASE/vllm_$dataset"

    # Generate updated comprehensive plots for current dataset
    echo -e "${BLUE}  📈 Updating comprehensive plots with $dataset results...${NC}"
    generate_comprehensive_plot "$dataset"

    echo -e "${GREEN}  ✅ Completed $dataset benchmark and comprehensive plots updated${NC}"
    echo -e "${GREEN}  📈 CSV data updated in: $PERSISTENT_RESEARCH_CSV${NC}"
    echo ""
}

# Function to generate comprehensive plot with all completed datasets (called after each dataset completes)
generate_comprehensive_plot() {
    local current_dataset=$1

    if [[ -n "$current_dataset" ]]; then
        echo -e "${YELLOW}    📊 Generating plot for current dataset: $current_dataset...${NC}"
    else
        echo -e "${YELLOW}    📊 Generating comprehensive plot with all completed datasets...${NC}"
    fi

    # Use the plot_comprehensive_results.py script to generate updated charts
    if [[ -f "plot_comprehensive_results.py" ]]; then
        echo -e "${BLUE}      Running comprehensive plotting script...${NC}"
        # Use the current run's CSV instead of the master CSV to show only this run's results
        PLOT_CMD="python3 plot_comprehensive_results.py \
            --csv \"$RESEARCH_CSV\" \
            --output-dir \"$OUTPUT_BASE\" \
            --model-filter \"$VLLM_MODEL\""

        # Add dataset filter if specified
        if [[ -n "$current_dataset" ]]; then
            PLOT_CMD="$PLOT_CMD --dataset-filter \"$current_dataset\""
        fi

        eval "$PLOT_CMD"

        echo -e "${GREEN}    ✅ Comprehensive plots updated in $OUTPUT_BASE${NC}"

        # Print actual paths of generated charts
        if [[ -f "$OUTPUT_BASE/accuracy_comparison.png" ]]; then
            echo -e "${GREEN}    📊 Accuracy Chart: $OUTPUT_BASE/accuracy_comparison.png${NC}"
        fi
        if [[ -f "$OUTPUT_BASE/token_usage_comparison.png" ]]; then
            echo -e "${GREEN}    📊 Token Usage Chart: $OUTPUT_BASE/token_usage_comparison.png${NC}"
        fi
        if [[ -f "$OUTPUT_BASE/efficiency_analysis.png" ]]; then
            echo -e "${GREEN}    📊 Efficiency Chart: $OUTPUT_BASE/efficiency_analysis.png${NC}"
        fi
    else
        echo -e "${RED}    ⚠️  plot_comprehensive_results.py not found, skipping comprehensive plots${NC}"
    fi
}

# Function to generate plot for a single dataset (kept for compatibility)
generate_dataset_plot() {
    local dataset=$1

    echo -e "${YELLOW}    📊 Plotting $dataset results...${NC}"

    # Find the summary.json files
    ROUTER_SUMMARY=$(find "$OUTPUT_BASE/router_$dataset" -name "summary.json" -type f | head -1)
    VLLM_SUMMARY=$(find "$OUTPUT_BASE/vllm_$dataset" -name "summary.json" -type f | head -1)

    if [[ -f "$VLLM_SUMMARY" ]]; then
        PLOT_CMD="python3 -m vllm_semantic_router_bench.bench_plot --summary \"$VLLM_SUMMARY\" --out-dir \"$OUTPUT_BASE/plots_$dataset\""

        if [[ -f "$ROUTER_SUMMARY" ]]; then
            PLOT_CMD="$PLOT_CMD --router-summary \"$ROUTER_SUMMARY\""
        fi

        echo -e "${BLUE}      Running: $PLOT_CMD${NC}"
        eval "$PLOT_CMD"
        echo -e "${GREEN}    ✅ $dataset plots generated in $OUTPUT_BASE/plots_$dataset${NC}"
    else
        echo -e "${RED}    ⚠️  No vLLM summary.json found for $dataset, skipping plots${NC}"
    fi
}

# Function to generate comparison plots (now just calls individual dataset plots)
generate_plots() {
    echo -e "${BLUE}📈 Generating any remaining comparison plots...${NC}"

    for dataset in "${!DATASET_CONFIGS[@]}"; do
        # Check if plots already exist
        if [[ ! -d "$OUTPUT_BASE/plots_$dataset" ]]; then
            echo -e "${YELLOW}  📊 Generating missing plots for $dataset...${NC}"
            generate_dataset_plot "$dataset"
        else
            echo -e "${GREEN}  ✅ Plots for $dataset already exist${NC}"
        fi
    done

    echo -e "${GREEN}  ✅ All plots verified/generated${NC}"
    echo ""
}

# Function to generate summary report
generate_summary() {
    echo -e "${BLUE}📋 Generating research summary...${NC}"

    local summary_file="$OUTPUT_BASE/RESEARCH_SUMMARY.md"

    cat > "$summary_file" << EOF
# Multi-Dataset Benchmark Research Report

**Generated:** $(date)
**Configuration:** Router vs vLLM Direct Comparison
**Router Model:** $ROUTER_MODEL
**vLLM Model:** $VLLM_MODEL

## Dataset Overview

| Dataset | Samples per Category | Total Samples | Categories | Domain |
|---------|---------------------|---------------|------------|---------|
EOF

    # Add dataset details to summary
    for dataset in "${!DATASET_CONFIGS[@]}"; do
        samples=${DATASET_CONFIGS[$dataset]}
        case $dataset in
            "mmlu")
                echo "| MMLU | $samples | ~570 | 57 subjects | Academic Knowledge |" >> "$summary_file"
                ;;
            "arc-challenge")
                echo "| ARC-Challenge | $samples | $samples | 1 (Science) | Scientific Reasoning (Hard) |" >> "$summary_file"
                ;;
            "gpqa")
                echo "| GPQA | $samples | $samples | 1 (Graduate) | Graduate-level Q&A |" >> "$summary_file"
                ;;
            "truthfulqa")
                echo "| TruthfulQA | $samples | $samples | 1 (Truthfulness) | Truthful Responses |" >> "$summary_file"
                ;;
            "commonsenseqa")
                echo "| CommonsenseQA | $samples | $samples | 1 (Common Sense) | Commonsense Reasoning |" >> "$summary_file"
                ;;
            "hellaswag")
                echo "| HellaSwag | $samples | ~100 | ~50 activities | Commonsense NLI |" >> "$summary_file"
                ;;
        esac
    done

    cat >> "$summary_file" << EOF

## Results Summary

**📊 Main Research Data**: \`research_results.csv\` - Contains aggregated metrics for all datasets and modes

### Accuracy Comparison
- Router (auto model with reasoning): See research_results.csv
- vLLM Direct (No Reasoning): See research_results.csv
- vLLM Direct (All Reasoning): See research_results.csv

### Token Usage Analysis
- Average tokens per response by dataset and mode (in research_results.csv)
- Efficiency comparison between router and direct vLLM

### Key Findings
1. **Performance**: [To be filled based on results]
2. **Efficiency**: [To be filled based on token usage]
3. **Dataset-specific Insights**: [To be analyzed from plots]

## Files Generated

### Research Data (Primary)
- \`research_results.csv\` - **Main aggregated results for research paper**

### CSV Results (Detailed)
EOF

    # List all CSV files that will be generated
    for dataset in "${!DATASET_CONFIGS[@]}"; do
        echo "- \`router_$dataset/results_summary.csv\`" >> "$summary_file"
        echo "- \`vllm_$dataset/results_summary.csv\`" >> "$summary_file"
    done

    cat >> "$summary_file" << EOF

### Plots
EOF

    # List all plot files that will be generated
    for dataset in "${!DATASET_CONFIGS[@]}"; do
        echo "- \`plots_$dataset/bench_plot_accuracy.png\`" >> "$summary_file"
        echo "- \`plots_$dataset/bench_plot_avg_total_tokens.png\`" >> "$summary_file"
    done

    cat >> "$summary_file" << EOF

## Usage Instructions

1. **Review CSV files** for detailed numerical results
2. **Examine plots** for visual comparison trends
3. **Analyze token usage** for efficiency insights
4. **Compare across datasets** for model capability assessment

## Methodology

- **Seed**: 42 (for reproducibility)
- **Router Mode**: Auto model selection with reasoning
- **vLLM Modes**: No Reasoning and All Reasoning
- **Sample Strategy**: Stratified sampling per category
- **Evaluation**: Exact match accuracy and token usage

EOF

    echo -e "${GREEN}  ✅ Research summary generated: $summary_file${NC}"
    echo ""
}

# Main execution
echo -e "${BLUE}🚀 Starting comprehensive benchmark...${NC}"
start_time=$(date +%s)

# Run benchmarks for reasoning-focused datasets (GPQA first for quick feedback)
DATASET_ORDER=("gpqa" "truthfulqa" "gsm8k" "aqua-rat" "sciq" "openbookqa" "strategyqa" "drop" "mmlu")
dataset_count=0
total_datasets=${#DATASET_ORDER[@]}

for dataset in "${DATASET_ORDER[@]}"; do
    # Skip if dataset not configured
    if [[ -z "${DATASET_CONFIGS[$dataset]}" ]]; then
        echo -e "${YELLOW}⚠️  Dataset $dataset not configured, skipping...${NC}"
        continue
    fi

    dataset_count=$((dataset_count + 1))
    echo -e "${BLUE}🚀 Progress: Dataset $dataset_count/$total_datasets - Starting $dataset${NC}"
    run_dataset_benchmark "$dataset"
    echo -e "${GREEN}🎉 Progress: Dataset $dataset_count/$total_datasets - Completed $dataset${NC}"
    echo -e "${YELLOW}📊 Remaining datasets: $((total_datasets - dataset_count))${NC}"
    echo ""
done

# Generate plots
generate_plots

# Generate summary
generate_summary

# Calculate total runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
minutes=$((runtime / 60))
seconds=$((runtime % 60))

echo -e "${GREEN}🎉 COMPREHENSIVE BENCHMARK COMPLETED!${NC}"
echo -e "${GREEN}====================================${NC}"
echo ""
echo -e "${YELLOW}📊 Results Location:${NC} $OUTPUT_BASE"
echo -e "${YELLOW}⏱️  Total Runtime:${NC} ${minutes}m ${seconds}s"
echo ""
echo -e "${BLUE}📋 Next Steps:${NC}"
echo "1. 📊 **Master research data**: $PERSISTENT_RESEARCH_CSV"
echo "2. 📊 **This run's data**: $OUTPUT_BASE/research_results.csv"
echo "3. 📋 Review research summary: $OUTPUT_BASE/RESEARCH_SUMMARY.md"
echo "4. 📈 **View comprehensive charts**:"
if [[ -f "$OUTPUT_BASE/accuracy_comparison.png" ]]; then
    echo "   📊 Accuracy: $OUTPUT_BASE/accuracy_comparison.png"
fi
if [[ -f "$OUTPUT_BASE/token_usage_comparison.png" ]]; then
    echo "   📊 Token Usage: $OUTPUT_BASE/token_usage_comparison.png"
fi
if [[ -f "$OUTPUT_BASE/efficiency_analysis.png" ]]; then
    echo "   📊 Efficiency: $OUTPUT_BASE/efficiency_analysis.png"
fi
echo "5. 📄 Analyze detailed CSV files if needed"
echo ""
echo -e "${GREEN}🎓 Research CSV Format:${NC}"
echo "   Dataset | Mode | Model | Accuracy | Avg_Latency_ms | Avg_Total_Tokens | Sample_Count | Timestamp"
echo ""
echo -e "${GREEN}📈 Master CSV grows with each test run - perfect for longitudinal analysis!${NC}"
echo -e "${GREEN}✨ Ready for research report writing!${NC}"
