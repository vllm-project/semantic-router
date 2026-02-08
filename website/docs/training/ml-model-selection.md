# ML-Based Model Selection

This document covers the configuration and experimental results for ML-based model selection techniques implemented in the Semantic Router.

## Overview

ML-based model selection uses machine learning algorithms to intelligently route queries to the most appropriate LLM based on query characteristics and historical performance data. This approach provides significant improvements over random or single-model routing strategies.

### Supported Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **KNN** (K-Nearest Neighbors) | Quality-weighted voting among similar queries | High accuracy, diverse query types |
| **KMeans** | Cluster-based routing with efficiency optimization | Fast inference, balanced workloads |
| **SVM** (Support Vector Machine) | RBF kernel decision boundaries | Clear domain separation |

### Reference Papers

- [FusionFactory (arXiv:2507.10540)](https://arxiv.org/abs/2507.10540) - Query-level fusion via LLM routers
- [Avengers-Pro (arXiv:2508.12631)](https://arxiv.org/abs/2508.12631) - Performance-efficiency optimized routing

## Configuration

### Basic Configuration

Enable ML-based model selection in your `config.yaml`:

```yaml
# Enable ML model selection
model_selection:
  ml:
    enabled: true
    models_path: ".cache/ml-models"  # Path to trained model files

# Embedding model for query representation
embedding_models:
  qwen3_model_path: "models/mom-embedding-pro"  # Qwen3-Embedding-0.6B
```

### Per-Decision Algorithm Configuration

Configure different algorithms for different decision types:

```yaml
decisions:
  # Math queries - use KNN for quality-weighted selection
  - name: "math_decision"
    description: "Mathematics and quantitative reasoning"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "math"
    algorithm:
      type: "knn"
      knn:
        k: 5
    modelRefs:
      - model: "llama-3.2-1b"
      - model: "llama-3.2-3b"
      - model: "mistral-7b"

  # Coding queries - use SVM for clear boundaries
  - name: "code_decision"
    description: "Programming and software development"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "computer science"
    algorithm:
      type: "svm"
      svm:
        kernel: "rbf"
        gamma: 1.0
    modelRefs:
      - model: "codellama-7b"
      - model: "llama-3.2-3b"
      - model: "mistral-7b"

  # General queries - use KMeans for efficiency
  - name: "general_decision"
    description: "General knowledge queries"
    priority: 50
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "other"
    algorithm:
      type: "kmeans"
      kmeans:
        num_clusters: 8
    modelRefs:
      - model: "llama-3.2-1b"
      - model: "llama-3.2-3b"
      - model: "mistral-7b"
```

### Algorithm Parameters

#### KNN Parameters

```yaml
algorithm:
  type: "knn"
  knn:
    k: 5  # Number of neighbors (default: 5)
```

#### KMeans Parameters

```yaml
algorithm:
  type: "kmeans"
  kmeans:
    num_clusters: 8  # Number of clusters (default: 8)
```

#### SVM Parameters

```yaml
algorithm:
  type: "svm"
  svm:
    kernel: "rbf"   # Kernel type: rbf, linear (default: rbf)
    gamma: 1.0      # RBF kernel gamma (default: 1.0)
```

## Experimental Results

### Benchmark Setup

- **Test queries**: 109 queries across multiple domains
- **Models evaluated**: 4 LLMs (codellama-7b, llama-3.2-1b, llama-3.2-3b, mistral-7b)
- **Embedding model**: Qwen3-Embedding-0.6B (1024 dimensions)
- **Validation data**: Real benchmark queries with ground truth performance scores

### Performance Comparison

| Strategy | Avg Quality | Avg Latency | Best Model % |
|----------|-------------|-------------|--------------|
| **Oracle (best possible)** | 0.495 | 10.57s | 100.0% |
| **KMEANS Selection** | 0.252 | 20.23s | 23.9% |
| Always llama-3.2-3b | 0.242 | 25.08s | 15.6% |
| **SVM Selection** | 0.233 | 25.83s | 14.7% |
| Always mistral-7b | 0.215 | 70.08s | 13.8% |
| Always llama-3.2-1b | 0.212 | 3.65s | 26.6% |
| **KNN Selection** | 0.196 | 36.62s | 13.8% |
| Random Selection | 0.174 | 40.12s | 9.2% |
| Always codellama-7b | 0.161 | 53.78s | 4.6% |

### ML Routing Benefit Over Random Selection

| Algorithm | Quality Improvement | Best Model Selection |
|-----------|---------------------|---------------------|
| **KMEANS** | **+45.5%** | 2.6x more often |
| **SVM** | **+34.4%** | 1.6x more often |
| **KNN** | **+13.1%** | 1.5x more often |

### Key Findings

1. **All ML methods outperform random selection** - Significant quality improvements across all algorithms
2. **KMEANS provides best quality** - 45% improvement over random with good latency
3. **SVM offers balanced performance** - 34% improvement with clear decision boundaries
4. **KNN provides diverse model selection** - Uses all available models based on query similarity

## Architecture

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         ONLINE INFERENCE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Request (model="auto")                                             │
│       ↓                                                             │
│  Generate Query Embedding (Qwen3, 1024-dim)                         │
│       ↓                                                             │
│  Add Category One-Hot (14-dim) → 1038-dim feature vector            │
│       ↓                                                             │
│  Decision Engine → Match decision by domain                         │
│       ↓                                                             │
│  Load ML Selector (KNN/KMeans/SVM from JSON)                        │
│       ↓                                                             │
│  Run Inference → Select best model                                  │
│       ↓                                                             │
│  Route to selected LLM endpoint                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Training Your Own Models

**Offline Training vs Online Inference:**

- **Offline Training**: Done in **Python** using scikit-learn for KNN, KMeans, and SVM
- **Online Inference**: Done in **Rust** using [Linfa](https://github.com/rust-ml/linfa) via `ml-binding`

This separation allows for flexible training with Python's rich ML ecosystem while maintaining high-performance inference in production with Rust.

### Prerequisites

```bash
cd src/training/ml_model_selection
pip install -r requirements.txt
```

### Option 1: Download Pretrained Models

```bash
python download_model.py \
  --output-dir ../../../.cache/ml-models \
  --repo-id abdallah1008/semantic-router-ml-models
```

### Option 2: Train Using Pre-Benchmarked Data from HuggingFace

We provide ready-to-use benchmark data on HuggingFace that you can use directly for training:

**HuggingFace Dataset:** [abdallah1008/ml-selection-benchmark-data](https://huggingface.co/datasets/abdallah1008/ml-selection-benchmark-data)

| File | Description |
|------|-------------|
| `benchmark_training_data.jsonl` | Pre-benchmarked data with 4 models (codellama-7b, llama-3.2-1b, llama-3.2-3b, mistral-7b) |
| `validation_benchmark_with_gt.jsonl` | Validation data with ground truth for testing |

```bash
# Download benchmark data
huggingface-cli download abdallah1008/ml-selection-benchmark-data \
  --repo-type dataset \
  --local-dir .cache/ml-models

# Train directly using the pre-benchmarked data
python train.py \
  --data-file .cache/ml-models/benchmark_training_data.jsonl \
  --output-dir models/
```

This is the fastest way to get started - no need to run your own LLM benchmarks!

### Option 3: Train with Your Own Data

#### Step 1: Prepare Input Data (JSONL Format)

Create a JSONL file with your queries. Each line must contain `query` and `category` fields:

```jsonl
{"query": "What is the derivative of x^2?", "category": "math", "ground_truth": "2x"}
{"query": "Write a Python function to sort a list", "category": "computer science", "ground_truth": "def sort(lst): return sorted(lst)"}
{"query": "Explain photosynthesis", "category": "biology", "ground_truth": "Process where plants convert sunlight to energy"}
{"query": "What are the legal requirements for a contract?", "category": "law"}
```

**Required fields:**

| Field | Type | Description |
|-------|------|-------------|
| `query` | string | The input query text |
| `category` | string | Domain category (see [VSR Categories](#vsr-categories)) |
| `ground_truth` | string | Expected answer (required for calculating performance/quality scores) |

**Recommended fields (for accurate performance scoring):**

| Field | Type | Description |
|-------|------|-------------|
| `metric` | string | Evaluation method - determines how performance is calculated |
| `choices` | string | For multiple choice questions - signals MC evaluation |

**Optional fields:**

| Field | Type | Description |
|-------|------|-------------|
| `task_name` | string | Task identifier for logging/tracking (e.g., "mmlu", "gsm8k") |

**Important: Metric Field**

Without `metric`, the benchmark uses **CEM (Conditional Exact Match)** as default, which may not accurately score:

- Math problems (use `metric: "GSM8K"` or `metric: "MATH"`)
- Multiple choice (use `metric: "em_mc"` or include `choices`)
- Code generation (use `metric: "code_eval"`)

For best results, always specify the appropriate `metric` for your question type.

**Multiple Choice Questions**

For multiple choice questions, include `choices` (can be the options as string) and set `ground_truth` to the correct letter:

```jsonl
{"query": "What is the capital of France?\nA) London\nB) Paris\nC) Berlin\nD) Rome", "category": "other", "ground_truth": "B", "choices": "London,Paris,Berlin,Rome"}
```

The benchmark script:

1. Detects multiple choice via `choices` field or `metric: "em_mc"`
2. Extracts the answer letter (A/B/C/D) from the model's response
3. Compares against `ground_truth` (the correct letter)

**Evaluation Metrics**

The `metric` field controls how performance is calculated:

| Metric | Description | Example ground_truth |
|--------|-------------|----------------------|
| `em_mc` | Multiple choice - extract letter | `"B"` |
| `GSM8K` | Math - extract number after `####` | `"explanation #### 42"` |
| `MATH` | LaTeX math - extract from `\boxed{}` | `"\\boxed{2x+1}"` |
| `f1_score` | Text overlap F1 score | `"Paris is the capital"` |
| `code_eval` | Run code assertions | `"['assert func(1)==2']"` |
| (default) | CEM - containment match | `"Paris"` |

**Ground Truth is Required for Training**

The `ground_truth` field is essential for training ML model selection. Without it, the system cannot calculate which model performed better on each query. The training process compares each LLM's response against `ground_truth` to compute performance scores.

#### Step 2: Configure Your LLM Endpoints (models.yaml)

Create a `models.yaml` file to configure your LLM endpoints with authentication:

```yaml
models:
  # Local Ollama model (no auth required)
  - name: llama-3.2-1b
    endpoint: http://localhost:11434/v1

  - name: llama-3.2-3b
    endpoint: http://localhost:11434/v1

  # OpenAI with API key from environment variable
  - name: gpt-4
    endpoint: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    max_tokens: 2048
    temperature: 0.0

  # HuggingFace with token
  - name: mistral-7b-hf
    endpoint: https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2
    api_key: ${HF_TOKEN}
    headers:
      Authorization: "Bearer ${HF_TOKEN}"

  # Custom API with Bearer token
  - name: custom-llm
    endpoint: https://api.custom.com/v1
    api_key: ${CUSTOM_API_KEY}
    headers:
      Authorization: "Bearer ${CUSTOM_API_KEY}"
      X-Custom-Header: "value"
    max_tokens: 1024
    temperature: 0.1

  # vLLM self-hosted
  - name: codellama-7b
    endpoint: http://vllm-server:8000/v1
    # No auth needed for local vLLM
```

#### Step 3: Run Benchmark

The benchmark script sends each query to all configured LLMs and measures:

**Performance (Quality Score 0-1):**

| Query Type | Scoring Method |
|------------|----------------|
| **Multiple Choice** (A/B/C/D) | Exact match of selected option vs `ground_truth` |
| **Numeric/Math** | Parse and compare numbers (tolerance-based) |
| **Text/Code** | F1 score between model response and `ground_truth` |
| **Exact Match** | Binary 1.0 if exact match, 0.0 otherwise |

**Latency (Response Time):**

- Measured from request sent to response received (in seconds)
- Includes network latency + model inference time
- Used for efficiency weighting: `speed_factor = 1 / (1 + latency)`

**Output Format:**

The benchmark generates JSONL with one record per (query, model) pair:

```jsonl
{"query": "What is 2+2?", "category": "math", "model_name": "llama-3.2-1b", "response": "4", "ground_truth": "4", "performance": 1.0, "response_time": 0.523}
{"query": "What is 2+2?", "category": "math", "model_name": "llama-3.2-3b", "response": "The answer is 4", "ground_truth": "4", "performance": 0.85, "response_time": 1.234}
{"query": "What is 2+2?", "category": "math", "model_name": "mistral-7b", "response": "2+2=4", "ground_truth": "4", "performance": 0.92, "response_time": 2.156}
```

**Run Benchmark:**

```bash
# Using model config file (recommended)
python benchmark.py \
  --queries your_queries.jsonl \
  --model-config models.yaml \
  --output benchmark_output.jsonl \
  --concurrency 4 \
  --limit 500  # Optional: limit number of queries for testing

# Or using simple model list (all same endpoint)
python benchmark.py \
  --queries your_queries.jsonl \
  --models llama-3.2-1b,llama-3.2-3b,mistral-7b \
  --endpoint http://localhost:11434/v1 \
  --output benchmark_output.jsonl
```

**benchmark.py Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--queries` | (required) | Path to input JSONL file with queries |
| `--model-config` | None | Path to models.yaml with endpoint configs |
| `--models` | None | Comma-separated model names (alternative to --model-config) |
| `--endpoint` | `http://localhost:8000/v1` | API endpoint (used with --models) |
| `--output` | `benchmark_output.jsonl` | Output file path |
| `--concurrency` | `4` | Number of parallel requests to LLMs |
| `--limit` | None | Limit number of queries to process |
| `--max-tokens` | `1024` | Maximum tokens in LLM response |
| `--temperature` | `0.0` | Temperature for generation (0.0 = deterministic) |

**Concurrency Parameter**

The `--concurrency` parameter controls how many requests are sent to LLMs in parallel:

- **Higher values** (8-16): Faster benchmarking, but may overwhelm local models
- **Lower values** (1-2): Slower but safer for resource-constrained environments
- **Recommended**: Start with 4, increase if your LLM server can handle more

For Ollama on a single GPU, use `--concurrency 2-4`. For cloud APIs (OpenAI, HuggingFace), you can use `--concurrency 8-16`.

#### Step 4: Train ML Models

```bash
python train.py \
  --data-file benchmark_output.jsonl \
  --output-dir models/
```

### train.py Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-file` | (required) | Path to JSONL benchmark data |
| `--output-dir` | `models/` | Directory to save trained model JSON files |
| `--embedding-model` | `qwen3` | Embedding model: `qwen3`, `gte`, `mpnet`, `e5`, `bge` |
| `--cache-dir` | `.cache/` | Cache directory for embeddings |
| `--knn-k` | `5` | Number of neighbors for KNN |
| `--kmeans-clusters` | `8` | Number of clusters for KMeans |
| `--svm-kernel` | `rbf` | SVM kernel: `rbf`, `linear` |
| `--svm-gamma` | `1.0` | SVM gamma for RBF kernel |
| `--quality-weight` | `0.9` | Quality vs speed weight (0=speed, 1=quality) |
| `--batch-size` | `32` | Batch size for embedding generation |
| `--device` | `cpu` | Device: `cpu`, `cuda`, `mps` |
| `--limit` | None | Limit number of training samples |

**Examples:**

```bash
# Train with custom KNN k value
python train.py \
  --data-file benchmark.jsonl \
  --output-dir models/ \
  --knn-k 7

# Train with limited samples (for testing)
python train.py \
  --data-file benchmark.jsonl \
  --output-dir models/ \
  --limit 1000

# Train with GPU acceleration
python train.py \
  --data-file benchmark.jsonl \
  --output-dir models/ \
  --device cuda \
  --batch-size 64

# Train with custom algorithm parameters
python train.py \
  --data-file benchmark.jsonl \
  --output-dir models/ \
  --knn-k 10 \
  --kmeans-clusters 12 \
  --svm-kernel rbf \
  --svm-gamma 0.5 \
  --quality-weight 0.85
```

### VSR Categories

The system supports 14 domain categories. Use exact names (with spaces, not underscores):

```text
biology, business, chemistry, computer science, economics, engineering,
health, history, law, math, other, philosophy, physics, psychology
```

### Validate Trained Models

Run the Go validation script to verify ML routing benefit:

```bash
cd src/training/ml_model_selection

# Set library paths (WSL/Linux)
export LD_LIBRARY_PATH=$PWD/../../../candle-binding/target/release:$PWD/../../../ml-binding/target/release:$LD_LIBRARY_PATH

# Run validation
go run validate.go --qwen3-model /path/to/Qwen3-Embedding-0.6B
```

## Model Files

The trained models are stored as JSON files:

| File | Algorithm | Size |
|------|-----------|------|
| `knn_model.json` | K-Nearest Neighbors | ~2-10 MB |
| `kmeans_model.json` | KMeans Clustering | ~50 KB |
| `svm_model.json` | Support Vector Machine | ~1-5 MB |

These files are downloaded from HuggingFace or generated during training:

- **Models**: [abdallah1008/semantic-router-ml-models](https://huggingface.co/abdallah1008/semantic-router-ml-models)
- **Benchmark Data**: [abdallah1008/ml-selection-benchmark-data](https://huggingface.co/datasets/abdallah1008/ml-selection-benchmark-data)

## Best Practices

### Algorithm Selection Guide

| Use Case | Recommended Algorithm | Reason |
|----------|----------------------|--------|
| **Quality-critical tasks** | KNN (k=5) | Quality-weighted voting provides best accuracy |
| **High-throughput systems** | KMeans | Fast cluster lookup, good latency |
| **Domain-specific routing** | SVM | Clear decision boundaries between domains |
| **General purpose** | KMEANS | Best balance of quality and speed |

### Hyperparameter Tuning

1. **KNN k value**: Start with k=5, increase for smoother decisions
2. **KMeans clusters**: Match to number of distinct query patterns (8-16 typical)
3. **SVM gamma**: Use 1.0 for normalized embeddings, adjust based on data spread

### Feature Vector Composition

The ML models use a 1038-dimensional feature vector:

- **1024 dimensions**: Qwen3 semantic embedding
- **14 dimensions**: Category one-hot encoding (VSR domain categories)

```text
Feature Vector = [embedding(1024)] ⊕ [category_one_hot(14)]
```

## Troubleshooting

### Models Not Loading

```text
Error: pretrained model not found
```

Download models from HuggingFace:

```bash
cd src/training/ml_model_selection
python download_model.py --output-dir ../../../.cache/ml-models
```

### Low Selection Accuracy

1. Ensure embedding model matches training (Qwen3-Embedding-0.6B)
2. Verify category classification is working
3. Check that model names in config match training data

### Dimension Mismatch

```text
Error: embedding dimension mismatch
```

Ensure you're using the same embedding model for training and inference (Qwen3 produces 1024 dimensions).

## Next Steps

- [Training Overview](/docs/training/training-overview) - General training documentation
- [Model Performance Evaluation](/docs/training/model-performance-eval) - Detailed performance metrics
