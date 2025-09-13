# vLLM Semantic Router Benchmark Suite

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A comprehensive benchmark suite for evaluating **semantic router** performance against **direct vLLM** across multiple reasoning datasets. Perfect for researchers and developers working on LLM routing, evaluation, and performance optimization.

## 🎯 Key Features

- **6 Major Reasoning Datasets**: MMLU-Pro, ARC, GPQA, TruthfulQA, CommonsenseQA, HellaSwag
- **Router vs vLLM Comparison**: Side-by-side performance evaluation
- **Multiple Evaluation Modes**: NR (neutral), XC (explicit CoT), NR_REASONING (auto-reasoning)
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

# List available datasets
vllm-semantic-router-bench list-datasets

# Run comprehensive multi-dataset benchmark
vllm-semantic-router-bench comprehensive
```

### Python API

```python
from vllm_semantic_router_bench import DatasetFactory, list_available_datasets

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

### Example Output Structure

```
results/
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
from vllm_semantic_router_bench import DatasetInterface, Question, DatasetInfo

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
