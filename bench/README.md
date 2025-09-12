# Router vs Direct vLLM Benchmark Commands

## 🚀 Quick One-Liner Commands

### Basic Comparison (ARC dataset, 3 samples per category)
```bash
# Router + Direct vLLM comparison
cd bench && source ../.venv/bin/activate && \
python3 router_reason_bench_multi_dataset.py --dataset arc --samples-per-category 3 --run-router --router-models auto --output-dir results/router_test && \
python3 router_reason_bench_multi_dataset.py --dataset arc --samples-per-category 3 --run-vllm --vllm-endpoint http://127.0.0.1:8000/v1 --vllm-models openai/gpt-oss-20b --vllm-exec-modes NR XC --output-dir results/vllm_test
```

### Comprehensive Script (Recommended)
```bash
cd bench && ./benchmark_comparison.sh arc 5
```

## 📋 Command Breakdown

### Router Evaluation (via Envoy)
- **Endpoint**: `http://127.0.0.1:8801/v1` (Envoy proxy)
- **Model**: `auto` (router decides which model to use)
- **API Key**: `1234` (default)
- **Purpose**: Tests the semantic router's routing decisions

```bash
python3 router_reason_bench_multi_dataset.py \
    --dataset arc \
    --samples-per-category 5 \
    --run-router \
    --router-endpoint http://127.0.0.1:8801/v1 \
    --router-api-key 1234 \
    --router-models auto
```

### Direct vLLM Evaluation
- **Endpoint**: `http://127.0.0.1:8000/v1` (direct vLLM)
- **Model**: `openai/gpt-oss-20b` (specific model)
- **API Key**: `1234` (default)
- **Modes**: 3 realistic scenarios (NR, XC, NR_REASONING)
- **Purpose**: Tests the raw model performance with scientific controls

```bash
python3 router_reason_bench_multi_dataset.py \
    --dataset arc \
    --samples-per-category 5 \
    --run-vllm \
    --vllm-endpoint http://127.0.0.1:8000/v1 \
    --vllm-api-key 1234 \
    --vllm-models openai/gpt-oss-20b
```

## 🎯 Available Datasets

- `arc` - AI2 Reasoning Challenge (both Easy + Challenge)
- `arc-easy` - ARC Easy questions only
- `arc-challenge` - ARC Challenge questions only  
- `mmlu` / `mmlu-pro` - MMLU-Pro dataset (14 categories)
- `gpqa` / `gpqa-main` - GPQA Main dataset (graduate-level)
- `gpqa-extended` - GPQA Extended dataset  
- `gpqa-diamond` - GPQA Diamond dataset (highest quality)
- `truthfulqa` - TruthfulQA dataset (6 categories, tests truthfulness)
- `commonsenseqa` - CommonsenseQA dataset (9 categories, tests reasoning)
- `hellaswag` - HellaSwag dataset (192 categories, tests commonsense)

## 📊 Example Usage

```bash
# Quick test with ARC
./benchmark_comparison.sh arc 3

# Comprehensive test with MMLU
./benchmark_comparison.sh mmlu 10

# Challenge questions only
./benchmark_comparison.sh arc-challenge 5
```

## 📈 Output Analysis

The script will create timestamped results in `results/comparison_YYYYMMDD_HHMMSS/`:
- Router results: `*router*auto*/`
- vLLM results: `*vllm*gpt-oss*/`
- **Comparison plots**: `plots/` directory with visual comparisons
- Each contains `summary.json` and `detailed_results.csv`

### 📊 Generated Visualizations
- `plots/bench_plot_accuracy.png` - Accuracy comparison by category
- `plots/bench_plot_avg_response_time.png` - Response time comparison
- `plots/bench_plot_avg_total_tokens.png` - Token usage comparison
- PDF versions of all plots are also generated

Compare:
- **Accuracy**: Overall correctness
- **Latency**: Response time per question
- **Tokens**: Token usage efficiency
- **Mode Performance**: NR vs XC reasoning approaches
