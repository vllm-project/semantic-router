# ML Model Selection Training

Python-based training for KNN, KMeans, and SVM model selection algorithms.

> **📝 Important: Data and Model Locations**
>
> **Models** and **Data** are stored in separate directories:
>
> | Type | HuggingFace Repo | Local Path |
> |------|------------------|------------|
> | **Trained Models** | `abdallah1008/semantic-router-ml-models` | `.cache/ml-models/` |
> | **Benchmark Data** | `abdallah1008/ml-selection-benchmark-data` | `.cache/ml-models/` |
>
> **Files:**
>
> - Models: `knn_model.json`, `kmeans_model.json`, `svm_model.json`
> - Data: `validation_benchmark_with_gt.jsonl`, `benchmark_data.jsonl`
>
> These files are deployment-specific and should be generated based on your own LLMs, queries, and benchmarks.

## Overview

This module trains machine learning models for query-based LLM routing, implementing:

- **KNN (K-Nearest Neighbors)**: Quality-weighted voting among similar queries
- **KMeans**: Cluster-based routing with efficiency optimization
- **SVM (Support Vector Machine)**: RBF kernel decision boundaries

## Reference Papers

- [FusionFactory (arXiv:2507.10540)](https://arxiv.org/abs/2507.10540) - Query-level fusion via LLM routers
- [Avengers-Pro (arXiv:2508.12631)](https://arxiv.org/abs/2508.12631) - Performance-efficiency optimized routing

## Installation

```bash
cd src/training/ml_model_selection
pip install -r requirements.txt
```

## Quick Start

### Option 1: Download Pretrained Models from HuggingFace

```bash
# Download models to .cache/ml-models/ (repo root)
python download_model.py \
  --output-dir ../../../.cache/ml-models \
  --repo-id abdallah1008/semantic-router-ml-models

# Download to custom location
python download_model.py --output-dir my_models/
```

### Option 2: Train with Your Own Custom LLMs

If you have your own LLMs and want to train model selection for them:

```bash
# Step 1: Prepare queries file (JSONL with query, ground_truth, and optionally category)
cat > my_queries.jsonl << 'EOF'
{"query": "Write a Python function to sort a list", "ground_truth": "def sort_list(lst): return sorted(lst)", "category": "computer science"}
{"query": "What is the derivative of x^2?", "ground_truth": "2x", "category": "math"}
{"query": "Explain photosynthesis", "ground_truth": "Process where plants convert sunlight to energy", "category": "biology"}
EOF

# Step 2: Benchmark your LLMs (generates performance + response_time)
# Category is preserved from input - no separate step needed!
python benchmark.py \
  --queries my_queries.jsonl \
  --models my-llm-v1,my-llm-v2,my-llm-v3 \
  --endpoint http://localhost:8000/v1 \
  --output benchmark_output.jsonl

# Step 3: Train models directly (category already in output)
python train.py \
  --data-file benchmark_output.jsonl \
  --output-dir models/

# Step 4: (Optional) Upload to HuggingFace
python upload_model.py --model-dir models/ --repo-id your-org/your-models
```

### Option 2b: Using Ollama Models

```bash
# Create models.yaml for Ollama
cat > models.yaml << 'EOF'
models:
  - name: llama3.2:1b
    endpoint: http://localhost:11434/v1
  - name: llama3.2:3b
    endpoint: http://localhost:11434/v1
  - name: mistral:7b
    endpoint: http://localhost:11434/v1
  - name: codellama:7b
    endpoint: http://localhost:11434/v1
EOF

# Run benchmark with config file
python benchmark.py \
  --queries my_queries.jsonl \
  --model-config models.yaml \
  --output benchmark_output.jsonl \
  --concurrency 4

# Train
python train.py --data-file benchmark_output.jsonl --output-dir models/
```

### Option 3: Train from Existing Benchmark Data

```bash
# First, download training data from HuggingFace (NOT in repo!)
python download_model.py --output-dir data/ --file benchmark_training_data.jsonl

# Train from downloaded benchmark data
python train.py \
  --data-file data/benchmark_training_data.jsonl \
  --output-dir models/

# Train with custom embedding model
python train.py \
  --data-file data.jsonl \
  --output-dir models/ \
  --embedding-model bge
```

> **Note:** The file `benchmark_training_data.jsonl` is hosted on HuggingFace, not in this repository.
> It must be downloaded first using `download_model.py`.

## Complete Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: PREPARE QUERIES                                            │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  Create queries.jsonl with test queries and ground truth:           │
│                                                                     │
│  {"query": "Write a function to sort a list", "ground_truth": ...}  │
│  {"query": "What is 2+2?", "ground_truth": "4"}                     │
│                                                                     │
│  Optional fields: task_name, metric, choices                        │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: BENCHMARK YOUR LLMS                                        │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  python benchmark.py \                                              │
│    --queries queries.jsonl \                                        │
│    --models llm-v1,llm-v2,llm-v3 \                                  │
│    --endpoint http://localhost:8000/v1 \                            │
│    --output benchmark_output.jsonl                                  │
│                                                                     │
│  This runs each query against ALL your LLMs and measures:           │
│  • performance (accuracy vs ground_truth, 0-1)                      │
│  • response_time (latency in seconds)                               │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: TRAIN ML MODELS                                            │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  python train.py \                                                  │
│    --data-file benchmark_output.jsonl \                             │
│    --output-dir models/                                             │
│                                                                     │
│  Note: If input queries had 'category' field, it's preserved in     │
│  benchmark output - no separate category step needed!               │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: UPLOAD TO HUGGINGFACE (Optional)                           │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  export HF_TOKEN=your_token                                         │
│  python upload_model.py --model-dir models/                         │
│                                                                     │
│  Models will be available at your HuggingFace repository            │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5: INFERENCE (Go/Rust)                                        │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  Router loads pretrained models:                                    │
│  • LoadPretrainedSelector("knn_model.json")                         │
│  • Models loaded via ml_binding.KNNFromJSON()                       │
│                                                                     │
│  Runtime selection:                                                 │
│  • Generate query embedding (Qwen3, 1024-dim)                       │
│  • Add category one-hot (14-dim) → 1038-dim feature vector          │
│  • Call selector.Select(embedding)                                  │
│  • Route to selected model                                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Command Line Options

### benchmark.py

**Key Features:**

- **Automatic deduplication**: Extracts unique queries from input (supports existing training data files)
- **Category preservation**: If input has `category` field, it's preserved in output (no separate step needed)
- **Flexible input**: Accepts both simple queries and full training data format

```
--queries           Path to JSONL input file (required)
                    Supports: {"query": "...", "category": "...", "ground_truth": "..."}
                    Also supports existing training data with model_name, performance fields
                    Automatically deduplicates and extracts unique queries

Model specification (one required):
--models            Comma-separated list of model names (all use same endpoint)
--model-config      Path to YAML config file (supports different endpoints/auth per model)

For --models:
--endpoint          OpenAI-compatible API endpoint (default: http://localhost:8000/v1)
--api-key           API key for the endpoint (uses LLM_API_KEY or OPENAI_API_KEY env vars)
--max-tokens        Maximum tokens in response (default: 1024)
--temperature       Temperature for generation (default: 0.0)

General:
--output            Output file path (default: benchmark_output.jsonl)
--concurrency       Number of concurrent requests (default: 4)
--limit             Limit number of queries to process (for testing)
--no-progress       Disable progress bar
```

**Example with existing training data:**

```bash
# Use existing training data - extracts unique queries, preserves categories
python benchmark.py \
  --queries existing_training_data.jsonl \
  --model-config models.yaml \
  --output benchmark_output.jsonl \
  --limit 500  # Test with 500 queries first
```

#### Model Config File (models.yaml)

For models with different endpoints or authentication, use a YAML config:

```yaml
models:
  # Local model (no auth)
  - name: llama-3.2-1b
    endpoint: http://localhost:8000/v1

  # OpenAI with API key from environment variable
  - name: gpt-4
    endpoint: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}

  # Custom API with Bearer token
  - name: custom-llm
    endpoint: https://api.custom.com/v1
    headers:
      Authorization: Bearer ${CUSTOM_TOKEN}
    max_tokens: 2048
    temperature: 0.1
```

See `models.example.yaml` for more examples.

### train.py

```
--data-file         Path to JSONL data file (downloads from HuggingFace if not specified)
--output-dir        Directory to save trained models (default: models/)
--embedding-model   Embedding model: qwen3, gte, mpnet, e5, bge (default: qwen3)
--cache-dir         Cache directory for downloads and embeddings
--knn-k             Number of neighbors for KNN (default: 5)
--kmeans-clusters   Number of clusters for KMeans (default: 8)
--svm-kernel        SVM kernel type: rbf, linear (default: rbf)
--svm-gamma         SVM gamma parameter for RBF kernel (default: 1.0)
--quality-weight    Quality weight for best model selection (default: 0.9)
--batch-size        Batch size for embedding generation (default: 32)
--device            Device for embedding model: cpu, cuda, mps (default: cpu)
```

### download_model.py

```
--output-dir        Directory to save downloaded models (default: models/)
--repo-id           HuggingFace repository ID (default: vllm-project/semantic-router-ml-models)
--file              Download a single file instead of all models
--token             HuggingFace token (uses HF_TOKEN env var if not provided)
```

### upload_model.py

```
--model-dir         Directory containing trained models (required)
--repo-id           HuggingFace repository ID (default: vllm-project/semantic-router-ml-models)
--private           Create private repository
--token             HuggingFace token (uses HF_TOKEN env var if not provided)
```

### validate.go (Go) - **Primary Validation Tool**

Validates that ML routing provides benefit over baselines using the **actual production Go/Rust code**.

This uses the same inference path as production:

- **Embeddings**: Qwen3-Embedding-0.6B via `candle-binding` (Rust)
- **ML Inference**: KNN/KMeans/SVM via `ml-binding` → **Linfa** (Rust)

**Automatically downloads from HuggingFace:**

- **Models**: `abdallah1008/semantic-router-ml-models` → `.cache/ml-models/`
  - `knn_model.json`, `kmeans_model.json`, `svm_model.json`
- **Benchmark Data**: `abdallah1008/ml-selection-benchmark-data` → `.cache/ml-models/`
  - `validation_benchmark_with_gt.jsonl`

```
--data-file         Path to benchmark data JSONL file (optional - downloads from HF if not provided)
--models-dir        Directory for model files (default: .cache/ml-models)
--algorithm         Algorithm to validate: knn, kmeans, svm, all (default: all)
--test-split        Fraction of data to use for testing (default: 1.0 = all data)
--seed              Random seed for reproducibility (default: 42)
--models-repo       HuggingFace repo for models (default: abdallah1008/semantic-router-ml-models)
--data-repo         HuggingFace dataset repo for data (default: abdallah1008/ml-selection-benchmark-data)
--qwen3-model       Path to Qwen3 model (default: auto-download from HuggingFace)
--no-download       Skip HuggingFace download, use local files only
--no-embeddings     Skip embedding generation (use random vectors for testing)
```

**Example:**

```bash
# Run from the training directory
cd src/training/ml_model_selection

# Set library paths for Rust bindings (WSL/Linux)
export LD_LIBRARY_PATH=$PWD/../../../candle-binding/target/release:$PWD/../../../ml-binding/target/release:$LD_LIBRARY_PATH

# Run validation (downloads models automatically)
go run validate.go

# Run with specific Qwen3 model path
go run validate.go --qwen3-model /path/to/Qwen3-Embedding-0.6B

# Validate specific algorithm
go run validate.go --algorithm knn

# Use local data file (still downloads models)
go run validate.go --data-file my_benchmark.jsonl

# Skip downloads, use local files only
go run validate.go --no-download --data-file local.jsonl

# Quick test without embeddings (random vectors)
go run validate.go --no-embeddings
```

**Output:**

```
======================================================================
  ML Model Selection Validation (Production Go/Rust Code)
======================================================================
Downloading from HuggingFace...
  Downloaded knn_model.json
  Downloaded kmeans_model.json
  Downloaded svm_model.json
  Downloaded validation_benchmark_with_gt.jsonl

Data file:   .cache/ml-models/validation_benchmark_with_gt.jsonl
Models dir:  .cache/ml-models
Algorithm:   all
Test split:  100%

Loading benchmark data...
Loaded 109 test queries with 4 models
Models: codellama-7b, llama-3.2-1b, llama-3.2-3b, mistral-7b

Initializing Qwen3 embedding model...
Loaded Qwen3 embedding model: /path/to/models/Qwen3-Embedding-0.6B

Generating embeddings for test queries...
  Generated embeddings: 109/109

Loaded KNN selector from .cache/ml-models
Loaded KMEANS selector from .cache/ml-models
Loaded SVM selector from .cache/ml-models

Evaluating strategies...

Validation Results (109 test queries, 4 models)
======================================================================
Strategy                    Avg Quality  Avg Latency   Best Model %
----------------------------------------------------------------------
Oracle (best)                     0.495       10.57s         100.0%
KMEANS Selection                  0.252       20.23s          23.9%
Always llama-3.2-3b               0.242       25.08s          15.6%
SVM Selection                     0.233       25.83s          14.7%
Always mistral-7b                 0.215       70.08s          13.8%
Always llama-3.2-1b               0.212        3.65s          26.6%
KNN Selection                     0.196       36.62s          13.8%
Random Selection                  0.175       40.46s          13.8%
Always codellama-7b               0.161       53.78s           4.6%
======================================================================

ML Routing Benefit:
  - KMEANS Selection improves quality by +44.4% over random
  - KMEANS Selection selects best model 1.7x more often than random
  - SVM Selection improves quality by +33.3% over random
  - SVM Selection selects best model 1.1x more often than random
  - KNN Selection improves quality by +12.2% over random
  - KNN Selection selects best model 1.0x more often than random

Note: This validation uses the ACTUAL production Go/Rust selectors.
```

## Data Format

Input data should be in JSONL format with the following fields:

```json
{
  "query": "What is the capital of France?",
  "category": "other",
  "model_name": "llama-3.2-3b",
  "performance": 0.85,
  "response_time": 1.234
}
```

| Field | Description |
|-------|-------------|
| `query` | The input query text |
| `category` | Domain category (see VSRCategories below) |
| `model_name` | Which LLM model was used |
| `performance` | Quality score (0-1) |
| `response_time` | Latency in seconds |

### VSR Categories (14 domains)

The domain classifier uses exactly these 14 categories. **Use exact names with spaces** (not underscores):

| Category | Description |
|----------|-------------|
| `biology` | Life sciences, genetics, ecology |
| `business` | Business, management, marketing |
| `chemistry` | Chemistry, chemical reactions |
| `computer science` | Programming, algorithms, CS theory |
| `economics` | Economics, finance, markets |
| `engineering` | Engineering disciplines |
| `health` | Medical, healthcare, anatomy |
| `history` | Historical events, periods |
| `law` | Legal matters, regulations |
| `math` | Mathematics, calculus, algebra |
| `other` | General knowledge (catch-all) |
| `philosophy` | Philosophy, ethics, logic |
| `physics` | Physics, mechanics, relativity |
| `psychology` | Psychology, behavior, cognition |

> **⚠️ Important:** When configuring decisions in `values.yaml`, domain names must match
> these categories **exactly** (e.g., `computer science` not `computer_science`).

## Output Models

The training produces three model files:

| File | Algorithm | Description |
|------|-----------|-------------|
| `knn_model.json` | K-Nearest Neighbors | Quality-weighted voting among k similar queries |
| `kmeans_model.json` | KMeans Clustering | Cluster assignment with efficiency weight |
| `svm_model.json` | Support Vector Machine | RBF kernel decision boundaries |

These files are in JSON format compatible with the Rust inference code in `ml-binding`.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Training Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Benchmarking (benchmark.py) - NEW                               │
│     ├── Run queries against multiple LLM endpoints                  │
│     ├── Measure performance (accuracy) and response_time            │
│     └── Output JSONL with query, model_name, performance, latency   │
│                                                                      │
│  2. Category Classification (add_category_to_training_data.py)      │
│     ├── Call VSR's /api/v1/classify/intent for each query           │
│     └── Add category field (14 domains)                             │
│                                                                      │
│  3. Data Loading (data_loader.py)                                   │
│     ├── Download from HuggingFace or load local JSONL               │
│     └── Parse routing records with quality/latency                  │
│                                                                      │
│  4. Embedding Generation (embeddings.py)                            │
│     ├── Use Qwen3-Embedding-0.6B (1024-dim)                         │
│     └── Cache embeddings for reuse                                  │
│                                                                      │
│  5. Feature Engineering                                             │
│     ├── Query embedding (1024-dim from Qwen3)                       │
│     ├── Category one-hot (14-dim)                                   │
│     └── Feature vector (1038-dim = 1024 + 14)                       │
│                                                                      │
│  6. Model Training (models.py)                                      │
│     ├── KNN: Quality-weighted k-nearest neighbor voting             │
│     ├── KMeans: Cluster assignment with efficiency weight           │
│     └── SVM: RBF kernel with weighted training samples              │
│                                                                      │
│  7. Model Export                                                    │
│     └── JSON format compatible with Rust inference                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Weighting Formula

All algorithms use the same quality+efficiency weighting:

```
score = 0.9 × quality + 0.1 × speed_factor
speed_factor = 1 / (1 + normalized_latency)
```

This prioritizes response quality (90%) while considering efficiency (10%).

## E2E Testing

The E2E test profile automatically handles model loading:

1. **Check for local models** → Use if available
2. **Download from HuggingFace** → Try first
3. **Train locally** → Fallback if download fails

```bash
# Run E2E tests
make e2e-test E2E_PROFILE=ml-model-selection
```

## Troubleshooting

### Download Failed

```
✗ Download failed: Repository Not Found
```

The HuggingFace repository may not exist yet. Train models locally:

```bash
python train.py --data-file benchmark.jsonl --output-dir models/
```

### Missing Dependencies

```
ModuleNotFoundError: No module named 'sentence_transformers'
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### CUDA Out of Memory

Use CPU for embedding generation:

```bash
python train.py --device cpu --batch-size 16
```

### Slow Training

Cache embeddings to speed up subsequent runs:

```bash
python train.py --cache-dir .cache/
```
