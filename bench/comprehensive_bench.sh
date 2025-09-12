#!/bin/bash

# Comprehensive Multi-Dataset Benchmark Script for Research Report
# This script benchmarks all available datasets with reasonable sample sizes
# for statistical significance while maintaining manageable runtime.

set -e

# Configuration
VENV_PATH="../.venv"
ROUTER_ENDPOINT="http://127.0.0.1:8801/v1"
VLLM_ENDPOINT="http://127.0.0.1:8000/v1"
VLLM_MODEL="openai/gpt-oss-20b"
ROUTER_MODEL="auto"
OUTPUT_BASE="results/comprehensive_research_$(date +%Y%m%d_%H%M%S)"

# Single persistent CSV file for all research results
PERSISTENT_RESEARCH_CSV="results/research_results_master.csv"

# Dataset configurations (dataset_name:samples_per_category)
# Balanced for statistical significance vs runtime
declare -A DATASET_CONFIGS=(
    ["mmlu"]=10          # 57 subjects × 10 = 570 samples
    ["arc"]=15           # 1 category × 15 = 15 samples  
    ["gpqa"]=20          # 1 category × 20 = 20 samples
    ["truthfulqa"]=15    # 1 category × 15 = 15 samples
    ["commonsenseqa"]=20 # 1 category × 20 = 20 samples
    ["hellaswag"]=8      # ~50 activities × 8 = ~400 samples
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

# Initialize persistent research results CSV (create header only if file doesn't exist)
if [[ ! -f "$PERSISTENT_RESEARCH_CSV" ]]; then
    echo "Dataset,Mode,Model,Accuracy,Avg_Latency_ms,Avg_Total_Tokens,Sample_Count,Timestamp" > "$PERSISTENT_RESEARCH_CSV"
    echo -e "${GREEN}📊 Created new master research CSV: $PERSISTENT_RESEARCH_CSV${NC}"
else
    echo -e "${BLUE}📊 Using existing master research CSV: $PERSISTENT_RESEARCH_CSV${NC}"
fi

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
        local temp_file="/tmp/metrics_$dataset_$mode.txt"
        
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
            model_name = 'auto'
        else:
            model_name = 'openai/gpt-oss-20b'
        
        # For vLLM, we might have multiple modes (NR, NR_REASONING)
        if '$mode' == 'vllm' and 'mode' in df.columns:
            for mode_type in df['mode'].unique():
                mode_df = df[df['mode'] == mode_type]
                
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
                
                csv_line = f'$dataset,vLLM_{mode_type},{model_name},{mode_accuracy:.3f},{mode_latency:.1f},{mode_tokens:.1f},{mode_samples},$timestamp'
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
    
    # Router benchmark
    echo -e "${YELLOW}  🤖 Running router evaluation...${NC}"
    python3 -m vllm_semantic_router_bench.router_reason_bench_multi_dataset \
        --dataset "$dataset" \
        --samples-per-category "$samples" \
        --run-router \
        --router-endpoint "$ROUTER_ENDPOINT" \
        --router-models "$ROUTER_MODEL" \
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
        --seed 42
    
    # Extract and save vLLM metrics immediately
    extract_and_save_metrics "$dataset" "vllm" "$OUTPUT_BASE/vllm_$dataset"
    
    echo -e "${GREEN}  ✅ Completed $dataset benchmark${NC}"
    echo ""
}

# Function to generate comparison plots
generate_plots() {
    echo -e "${BLUE}📈 Generating comparison plots...${NC}"
    
    for dataset in "${!DATASET_CONFIGS[@]}"; do
        echo -e "${YELLOW}  📊 Plotting $dataset results...${NC}"
        
        python3 -m vllm_semantic_router_bench.bench_plot \
            --router-dir "$OUTPUT_BASE/router_$dataset" \
            --vllm-dir "$OUTPUT_BASE/vllm_$dataset" \
            --output-dir "$OUTPUT_BASE/plots_$dataset" \
            --dataset-name "$dataset"
    done
    
    echo -e "${GREEN}  ✅ All plots generated${NC}"
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
            "arc")
                echo "| ARC | $samples | $samples | 1 (Science) | Scientific Reasoning |" >> "$summary_file"
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
                echo "| HellaSwag | $samples | ~400 | ~50 activities | Commonsense NLI |" >> "$summary_file"
                ;;
        esac
    done

    cat >> "$summary_file" << EOF

## Results Summary

**📊 Main Research Data**: \`research_results.csv\` - Contains aggregated metrics for all datasets and modes

### Accuracy Comparison
- Router (auto model with reasoning): See research_results.csv
- vLLM Direct (NR mode): See research_results.csv
- vLLM Direct (NR_REASONING mode): See research_results.csv

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
- **vLLM Modes**: NR (neutral) and NR_REASONING (with reasoning)
- **Sample Strategy**: Stratified sampling per category
- **Evaluation**: Exact match accuracy and token usage

EOF

    echo -e "${GREEN}  ✅ Research summary generated: $summary_file${NC}"
    echo ""
}

# Main execution
echo -e "${BLUE}🚀 Starting comprehensive benchmark...${NC}"
start_time=$(date +%s)

# Run benchmarks for all datasets
for dataset in "${!DATASET_CONFIGS[@]}"; do
    run_dataset_benchmark "$dataset"
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
echo "4. 📈 Examine plots for visual insights"
echo "5. 📄 Analyze detailed CSV files if needed"
echo ""
echo -e "${GREEN}🎓 Research CSV Format:${NC}"
echo "   Dataset | Mode | Model | Accuracy | Avg_Latency_ms | Avg_Total_Tokens | Sample_Count | Timestamp"
echo ""
echo -e "${GREEN}📈 Master CSV grows with each test run - perfect for longitudinal analysis!${NC}"
echo -e "${GREEN}✨ Ready for research report writing!${NC}"
