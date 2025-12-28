# vLLM Semantic Router Benchmark Suite

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A comprehensive benchmark suite for evaluating **semantic router** performance against **direct vLLM** across multiple reasoning datasets. Perfect for researchers and developers working on LLM routing, evaluation, and performance optimization.

## 🎯 Key Features

- **6 Major Reasoning Datasets**: MMLU-Pro, ARC, GPQA, TruthfulQA, CommonsenseQA, HellaSwag
- **Router vs vLLM Comparison**: Side-by-side performance evaluation
- **Multiple Evaluation Modes**: NR (neutral), XC (explicit CoT), NR_REASONING (auto-reasoning)
- **Reasoning Mode Evaluation** (Issue #42): Dedicated standard vs reasoning mode comparison
- **Research-Ready Output**: CSV files and publication-quality plots
- **Dataset-Agnostic Architecture**: Easy to extend with new datasets
- **CLI Tools**: Simple command-line interface for common operations

## 🚀 Quick Start

### Installation

```bash
pip install vllm-semantic-router-bench
```

### Basic Usage

```bash
# Quick test on MMLU dataset
vllm-semantic-router-bench test --dataset mmlu --samples 5

# Full comparison between router and vLLM
vllm-semantic-router-bench compare --dataset arc --samples 10

# Reasoning mode evaluation (Issue #42)
vllm-semantic-router-bench reasoning-eval --datasets mmlu gpqa --samples 10

# List available datasets
vllm-semantic-router-bench list-datasets

# Run comprehensive multi-dataset benchmark
vllm-semantic-router-bench comprehensive
```

### Reasoning Mode Evaluation (Issue #42)

Dedicated benchmark comparing standard vs reasoning mode with key metrics:

```bash
# Run reasoning mode evaluation
reasoning-mode-eval --datasets mmlu gpqa truthfulqa --samples-per-category 10

# Or use the shell script
./reasoning_mode_eval.sh
```

**Key Metrics Evaluated:**

- **Response Correctness**: Accuracy on MMLU(-Pro) and non-MMLU test sets
- **Token Usage Ratio**: `completion_tokens / prompt_tokens`
- **Time per Output Token**: Response time efficiency metric (ms)

**Automated vSR Config Generation:**

The benchmark automatically generates vLLM Semantic Router (vSR) model configuration based on evaluation results:

```bash
# Generate vSR config with reasoning family specification
reasoning-mode-eval \
  --datasets mmlu gpqa \
  --model qwen3-14b \
  --reasoning-family qwen3 \
  --samples-per-category 20
```

**Output includes:**

- `vsr_model_config.yaml` - Ready-to-use YAML config snippet for `config/config.yaml`
- `vsr_model_config_recommendation.json` - Detailed performance analysis and recommendations
- Automatic recommendation based on accuracy vs. cost/latency trade-offs

**Example generated config:**

```yaml
model_config:
  qwen3-14b:
    reasoning_family: qwen3
```

**Supported reasoning families:**

- `qwen3` - For Qwen-3 models with `chat_template_kwargs`
- `deepseek` - For DeepSeek-R1 models with `thinking` parameter
- `gpt-oss` - For GPT-OSS models with `reasoning_effort`

### Python API

```python
from reasoning import DatasetFactory, list_available_datasets

# Load a dataset
factory = DatasetFactory()
dataset = factory.create_dataset("mmlu")
questions, info = dataset.load_dataset(samples_per_category=10)

print(f"Loaded {len(questions)} questions from {info.name}")
print(f"Categories: {info.categories}")
```

## 📊 Supported Datasets

| Dataset | Domain | Categories | Difficulty | CoT Support |
|---------|--------|------------|------------|-------------|
| **MMLU-Pro** | Academic Knowledge | 57 subjects | Undergraduate | ✅ |
| **ARC** | Scientific Reasoning | Science | Grade School | ❌ |
| **GPQA** | Graduate Q&A | Graduate-level | Graduate | ❌ |
| **TruthfulQA** | Truthfulness | Truthfulness | Hard | ❌ |
| **CommonsenseQA** | Common Sense | Common Sense | Hard | ❌ |
| **HellaSwag** | Commonsense NLI | ~50 activities | Moderate | ❌ |

## 🔧 Advanced Usage

### Custom Evaluation Script

```python
import subprocess
import sys

# Run detailed benchmark with custom parameters
cmd = [
    "router-bench",  # Main benchmark script
    "--dataset", "mmlu",
    "--samples-per-category", "20", 
    "--run-router", "--router-models", "auto",
    "--run-vllm", "--vllm-models", "openai/gpt-oss-20b",
    "--vllm-exec-modes", "NR", "NR_REASONING",
    "--output-dir", "results/custom_test"
]

subprocess.run(cmd)
```

### Plotting Results

```bash
# Generate plots from benchmark results
bench-plot --router-dir results/router_mmlu \
           --vllm-dir results/vllm_mmlu \
           --output-dir results/plots \
           --dataset-name "MMLU-Pro"
```

## 📈 Research Output

The benchmark generates research-ready outputs:

- **CSV Files**: Detailed per-question results and aggregated metrics
- **Master CSV**: Combined results across all test runs
- **Plots**: Accuracy and token usage comparisons
- **Summary Reports**: Markdown reports with key findings

### Generated Output Structure

**Note**: The following directory structure is created locally when you run the benchmark. These files are not committed to the repository.

```
results/  # Created locally when running benchmarks
├── research_results_master.csv          # Main research data
├── comparison_20250115_143022/
│   ├── router_mmlu/
│   │   └── detailed_results.csv
│   ├── vllm_mmlu/  
│   │   └── detailed_results.csv
│   ├── plots/
│   │   ├── accuracy_comparison.png
│   │   └── token_usage_comparison.png
│   └── RESEARCH_SUMMARY.md
└── reasoning_mode_eval/                  # Issue #42 evaluation results
    ├── reasoning_mode_eval_summary.json  # Full evaluation summary with all metrics
    ├── vsr_model_config.yaml             # Ready-to-use vSR config snippet
    ├── vsr_model_config_recommendation.json  # Detailed recommendation & analysis
    ├── REASONING_MODE_EVALUATION_REPORT.md   # Human-readable report
    ├── plots/
    │   ├── MMLU-Pro_overall_comparison.png
    │   ├── MMLU-Pro_category_accuracy.png
    │   ├── MMLU-Pro_token_usage_ratio.png
    │   └── MMLU-Pro_time_per_token.png
    └── MMLU-Pro/
        ├── detailed_results.csv
        ├── standard_mode_results.csv
        └── reasoning_mode_results.csv
```

## 🚀 Using Generated vSR Config in Production

After running the reasoning mode evaluation, integrate the generated configuration into your semantic-router deployment:

### 1. Review the Recommendation

```bash
# Check the detailed recommendation
cat results/reasoning_mode_eval/vsr_model_config_recommendation.json

# View the generated config
cat results/reasoning_mode_eval/vsr_model_config.yaml
```

### 2. Integrate into config.yaml

Copy the generated `model_config` section to your `config/config.yaml`:

```yaml
# config/config.yaml

model_config:
  qwen3-14b:
    reasoning_family: qwen3  # From generated config
    preferred_endpoints: ["endpoint1"]  # Optional: your endpoint configuration
```

### 3. Enable Reasoning for Categories (Optional)

To enable reasoning mode for specific categories, update your intelligent routing configuration:

```yaml
# config/config.yaml

default_reasoning_effort: "medium"  # or "low", "high"

# OR enable per-category
categories:
  - name: math
    reasoning_enabled: true  # Enable reasoning for complex math queries
  - name: casual
    reasoning_enabled: false  # Disable for casual conversations
```

### 4. End-to-End Pipeline Example

```bash
# 1. Run evaluation
reasoning-mode-eval \
  --datasets mmlu gpqa truthfulqa \
  --model qwen3-14b \
  --reasoning-family qwen3 \
  --endpoint http://your-vllm-server:8000/v1 \
  --samples-per-category 50

# 2. Review results
cat results/reasoning_mode_eval/REASONING_MODE_EVALUATION_REPORT.md

# 3. If recommendation is positive, merge generated config
cp results/reasoning_mode_eval/vsr_model_config.yaml config/model_config_addition.yaml

# 4. Update your main config.yaml with the new model_config section

# 5. Restart semantic-router with updated config
kubectl rollout restart deployment semantic-router  # For K8s
# OR
docker-compose restart semantic-router  # For Docker Compose
```

## 🛠️ Development

### Local Installation

```bash
git clone https://github.com/vllm-project/semantic-router
cd semantic-router/bench
pip install -e ".[dev]"
```

### Adding New Datasets

1. Create a new dataset implementation in `dataset_implementations/`
2. Inherit from `DatasetInterface`
3. Register in `dataset_factory.py`
4. Add tests and documentation

```python
from reasoning import DatasetInterface, Question, DatasetInfo

class MyDataset(DatasetInterface):
    def load_dataset(self, **kwargs):
        # Implementation here
        pass
    
    def format_prompt(self, question, style="plain"):
        # Implementation here  
        pass
```

## 📋 Requirements

- Python 3.8+
- OpenAI API access (for model evaluation)
- Hugging Face account (for dataset access)
- 4GB+ RAM (for larger datasets)

### Dependencies

- `openai>=1.0.0` - OpenAI API client
- `datasets>=2.14.0` - Hugging Face datasets
- `pandas>=1.5.0` - Data manipulation
- `matplotlib>=3.5.0` - Plotting
- `seaborn>=0.11.0` - Advanced plotting
- `tqdm>=4.64.0` - Progress bars

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Common Contributions

- Adding new datasets
- Improving evaluation metrics
- Enhancing visualization
- Performance optimizations
- Documentation improvements

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Documentation**: https://vllm-semantic-router.com
- **GitHub**: https://github.com/vllm-project/semantic-router
- **Issues**: https://github.com/vllm-project/semantic-router/issues
- **PyPI**: https://pypi.org/project/vllm-semantic-router-bench/

## 📞 Support

- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive guides and API reference
- **Community**: Join our discussions and get help from other users

---

**Made with ❤️ by the vLLM Semantic Router Team**
