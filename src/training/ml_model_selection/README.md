# ML Model Selection Training

Python-based training for KNN, KMeans, and SVM model selection algorithms.

> **📝 Important: Training Data Location**
>
> The training data file `benchmark_training_data.jsonl` is **NOT included in this repository**.
> It must be downloaded from HuggingFace Hub:
>
> ```bash
> python download_model.py --output-dir models/ --file benchmark_training_data.jsonl
> ```
>
> Or download directly from: https://huggingface.co/vllm-project/semantic-router-ml-models
>
> This file is deployment-specific and should be generated based on your own LLMs, queries, and benchmarks.

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
# Download all models
python download_model.py --output-dir models/

# Download to specific location
python download_model.py \
  --output-dir ../../semantic-router/pkg/modelselection/data/trained_models \
  --repo-id abdallah1008/semantic-router-ml-models
```

### Option 2: Train with Your Own Custom LLMs

If you have your own LLMs and want to train model selection for them:

```bash
# Step 1: Prepare queries file (JSONL with query and ground_truth)
cat > my_queries.jsonl << 'EOF'
{"query": "Write a Python function to sort a list", "ground_truth": "def sort_list(lst): return sorted(lst)"}
{"query": "What is the derivative of x^2?", "ground_truth": "2x"}
{"query": "Explain photosynthesis", "ground_truth": "Process where plants convert sunlight to energy"}
EOF

# Step 2: Benchmark your LLMs (generates performance + response_time)
python benchmark.py \
  --queries my_queries.jsonl \
  --models my-llm-v1,my-llm-v2,my-llm-v3 \
  --endpoint http://localhost:8000/v1 \
  --output benchmark_output.jsonl

# Step 3: Add categories using VSR classifier (requires VSR running)
python ../../../src/semantic-router/pkg/modelselection/data/add_category_to_training_data.py \
  --input benchmark_output.jsonl \
  --output benchmark_with_category.jsonl \
  --vsr-url http://localhost:8080

# Step 4: Train models
python train.py \
  --data-file benchmark_with_category.jsonl \
  --output-dir models/

# Step 5: (Optional) Upload to HuggingFace
python upload_model.py --model-dir models/ --repo-id your-org/your-models
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
│  STEP 3: ADD CATEGORIES (VSR Classifier)                            │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  # Start VSR router first, then:                                    │
│  python add_category_to_training_data.py \                          │
│    --input benchmark_output.jsonl \                                 │
│    --output benchmark_with_category.jsonl \                         │
│    --vsr-url http://localhost:8080                                  │
│                                                                     │
│  Adds 'category' field (math, physics, computer science, etc.)      │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4: TRAIN ML MODELS                                            │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  python train.py \                                                  │
│    --data-file benchmark_with_category.jsonl \                      │
│    --output-dir models/                                             │
│                                                                     │
│  Training process:                                                  │
│  1. Load benchmark data                                             │
│  2. Generate 1024-dim embeddings (Qwen3)                            │
│  3. Create feature vectors (embedding + category one-hot = 1038)    │
│  4. Train KNN, KMeans, SVM with quality+speed weighting             │
│  5. Save models as JSON files                                       │
│                                                                     │
│  Output: knn_model.json, kmeans_model.json, svm_model.json          │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5: UPLOAD TO HUGGINGFACE (Optional)                           │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  export HF_TOKEN=your_token                                         │
│  python upload_model.py --model-dir models/                         │
│                                                                     │
│  Models will be available at your HuggingFace repository            │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 6: INFERENCE (Go/Rust)                                        │
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

```
--queries           Path to JSONL file with queries (required)

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
--no-progress       Disable progress bar
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
