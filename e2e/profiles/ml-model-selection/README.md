# ML-Based Model Selection E2E Profile

> **This profile demonstrates ML-based model selection using pretrained models from HuggingFace**
>
> **📥 Pretrained Models Downloaded Automatically**
>
> The E2E test automatically downloads pretrained ML models (KNN, KMeans, SVM, MLP) from HuggingFace
> during setup. No local training or Python virtual environment setup is required.
>
> | Type | HuggingFace Repo | Local Path |
> |------|------------------|------------|
> | **Trained Models** | `abdallah1008/semantic-router-ml-models` | `.cache/ml-models/` |
> | **Benchmark Data** | `abdallah1008/ml-selection-benchmark-data` | `.cache/ml-models/` |
>
> **Model Files:** `knn_model.json`, `kmeans_model.json`, `svm_model.json`, `mlp_model.json`
> **Data Files:** `validation_benchmark_with_gt.jsonl`

This profile demonstrates how to use pretrained ML models for intelligent model selection at runtime, implementing concepts from FusionFactory and Avengers-Pro papers.

**Production Ready:** This profile deploys the full production stack including:

- **Envoy Gateway** - Gateway API implementation
- **Envoy AI Gateway** - AI-specific CRDs and routing
- **gRPC ExtProc** - Semantic router as external processor
- **Custom AIGatewayRoute** - Routes `x-selected-model` headers to mock-llm backend

The profile uses custom gateway resources in `gateway-resources/` that match the semantic-router's output headers.

## Reference Papers

- **FusionFactory** ([arXiv:2507.10540](https://arxiv.org/abs/2507.10540)) - Query-level fusion via LLM routers
- **Avengers-Pro** ([arXiv:2508.12631](https://arxiv.org/abs/2508.12631)) - Performance-efficiency optimized routing

## Complete E2E Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1: DOWNLOAD PRETRAINED MODELS (Automatic)                     │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  E2E profile setup automatically downloads from HuggingFace:        │
│                                                                     │
│  From: abdallah1008/semantic-router-ml-models                       │
│  To:   .cache/ml-models/                                            │
│    • knn_model.json - K-Nearest Neighbors model                     │
│    • kmeans_model.json - KMeans clustering model                    │
│    • svm_model.json - Support Vector Machine model                  │
│    • mlp_model.json - Multi-Layer Perceptron model (GPU)            │
│                                                                     │
│  From: abdallah1008/ml-selection-benchmark-data                     │
│  To:   .cache/ml-models/                                            │
│    • validation_benchmark_with_gt.jsonl - Validation data           │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2: ROUTER LOADS MODELS AT STARTUP                             │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  selection.Factory.CreateAll():                                     │
│    → Creates KNN, KMeans, SVM, MLP selectors                        │
│    → Loads pretrained models from JSON                              │
│    → Registers in selection.Registry                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3: RUNTIME MODEL SELECTION                                    │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  Request: POST /v1/chat/completions                                 │
│           { "model": "MoM", "messages": [...] }                     │
│                                                                     │
│      ↓                                                              │
│  Decision Engine matches: math_decision (algorithm: knn)            │
│      ↓                                                              │
│  getSelectionMethod() → returns MethodKNN                           │
│      ↓                                                              │
│  Registry.Get(MethodKNN) → MLSelectorAdapter                        │
│      ↓                                                              │
│  KNNSelector.Select():                                              │
│    1. Generate embedding for query                                  │
│    2. Find K nearest neighbors in training data                     │
│    3. Quality-weighted voting among neighbors                       │
│    4. Return model with highest score                               │
│      ↓                                                              │
│  Response: x-vsr-selected-model: deepseek-math                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    PRETRAINED MODELS (HuggingFace)            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  HuggingFace Hub         →    Download    →    Local Path    │
│                                                              │
│  abdallah1008/semantic-     (automatic)     .cache/ml-models/
│   router-ml-models                          ├── knn_model.json
│                                             ├── kmeans_model.json
│                                             ├── svm_model.json
│                                             └── mlp_model.json
│                                                              │
│  abdallah1008/ml-selection- (automatic)     .cache/ml-models/
│   benchmark-data                            └── validation_benchmark_with_gt.jsonl
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                    ONLINE INFERENCE                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Request (model="auto")                                      │
│       ↓                                                      │
│  Decision Engine → Matches "math_decision"                   │
│       ↓                                                      │
│  Algorithm Config: type="knn", k=5                           │
│       ↓                                                      │
│  KNN Selector → Finds similar queries → Weighted voting      │
│       ↓                                                      │
│  Selected Model: "deepseek-math"                             │
│       ↓                                                      │
│  Route to deepseek-math endpoint                             │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

### Run E2E Tests

The E2E test automatically downloads pretrained models from HuggingFace during setup.
No manual preparation is required.

```bash
# Run the E2E test - models are downloaded automatically
make e2e-test E2E_PROFILE=ml-model-selection
```

### Manual Model Download (Optional)

If you want to download models manually before running tests:

```bash
pip install huggingface-hub

cd src/training/model_selection/ml_model_selection

# Download trained models to .cache/ml-models/ (repo root)
python download_model.py \
  --output-dir ../../../.cache/ml-models \
  --repo-id abdallah1008/semantic-router-ml-models

# Models will be saved to:
# - <repo-root>/.cache/ml-models/knn_model.json
# - <repo-root>/.cache/ml-models/kmeans_model.json
# - <repo-root>/.cache/ml-models/svm_model.json
```

### 3. Verify Selection

The test sends queries and verifies:

- Decision matches expected category (e.g., `math_decision`)
- Selected model is one of the expected models
- Algorithm header shows `knn`, `kmeans`, `svm`, or `mlp`

### 4. Validate ML Models (Optional)

To validate that ML routing provides benefit over baselines, use the `validate.go` script:

```bash
# Run from the training directory
cd src/training/model_selection/ml_model_selection

# Set library paths for Rust bindings (WSL/Linux)
export LD_LIBRARY_PATH=$PWD/../../../candle-binding/target/release:$PWD/../../../ml-binding/target/release:$LD_LIBRARY_PATH

# Run validation (downloads models automatically)
go run validate.go

# Or with specific Qwen3 model path
go run validate.go --qwen3-model /path/to/Qwen3-Embedding-0.6B
```

This uses the **actual production inference path**:

- **Embeddings**: Qwen3-Embedding-0.6B via `candle-binding` (Rust)
- **ML Inference**: KNN/KMeans/SVM via `ml-binding` → **Linfa** (Rust), MLP via `candle-binding` → **Candle** (GPU)

**Expected Results (109 test queries, 4 models):**

| Algorithm | Avg Quality | Improvement over Random |
|-----------|-------------|------------------------|
| **MLP** (GPU) | 0.286 | +47.1% |
| **KMEANS** | 0.252 | +29.9% |
| **SVM** | 0.233 | +20.0% |
| **KNN** | 0.196 | +1.0% |
| Random | 0.194 | baseline |

Output shows ML routing improvement vs baselines (random, single-model).

## Configuration

### VSR Domain Categories (14 domains)

The domain classifier uses exactly these 14 categories. **Domain names must match exactly** (with spaces, not underscores):

```
biology, business, chemistry, computer science, economics, engineering,
health, history, law, math, other, philosophy, physics, psychology
```

### Decision with ML Algorithm

```yaml
decisions:
  - name: "math_decision"
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "math"     # Must match VSR category exactly
    algorithm:
      type: "knn"          # Options: knn, kmeans, svm, mlp
      knn:
        k: 5               # Number of neighbors
    modelRefs:
      - model: "deepseek-math"
      - model: "mistral-7b"
      - model: "llama-3.2-3b"

  - name: "code_decision"
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "computer science"  # Note: space, not underscore!
    algorithm:
      type: "svm"
    svm:
      kernel: "rbf"
    modelRefs:
      - model: "codellama-7b"
      - model: "mistral-7b"

  - name: "gpu_decision"
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "engineering"
    algorithm:
      type: "mlp"
      mlp:
        device: "cuda"     # Options: cpu, cuda, metal
    modelRefs:
      - model: "llama-3.2-3b"
      - model: "mistral-7b"
```

### ML Selector Configuration

```yaml
model_selection:
  ml:
    models_path: ".cache/ml-models"
    knn:
      k: 5
      pretrained_path: ".cache/ml-models/knn_model.json"
    svm:
      kernel: "rbf"
      gamma: 1.0
```

## Algorithms

| Algorithm | Best For | Key Feature |
|-----------|----------|-------------|
| **KNN** | Similar query matching | Quality-weighted voting (0.9q + 0.1e) |
| **KMeans** | Efficiency optimization | Cluster-based routing |
| **SVM** | Clear preferences | RBF kernel decision boundaries |
| **MLP** | GPU-enabled environments | Neural network with CUDA/Metal acceleration |

## Test Cases

This profile includes 25 test cases covering all 8 decision types across the 14 VSR domain categories.

| Query Type | Decision | Algorithm | Domain(s) |
|------------|----------|-----------|-----------|
| Math/calculus | `math_decision` | knn | `math` |
| Code/programming | `code_decision` | svm | `computer science` |
| Physics/chemistry/biology | `science_decision` | kmeans | `physics`, `chemistry`, `biology` |
| Medical/health | `health_decision` | knn | `health` |
| Engineering | `engineering_decision` | mlp | `engineering` |
| Business/economics | `business_decision` | knn | `business`, `economics` |
| History/philosophy/law | `humanities_decision` | knn | `history`, `philosophy`, `psychology`, `law` |
| General knowledge | `general_decision` | knn | `other` |

> **⚠️ Domain Name Format:** Domain names must use **spaces** (e.g., `computer science`), not underscores.

### Test Case Structure

```json
{
  "query": "Calculate the derivative of sin(x) * cos(x)",
  "decision": "math_decision",
  "expected_models": ["llama-3.2-1b", "llama-3.2-3b", "codellama-7b", "mistral-7b"],
  "algorithm": "knn"
}
```

### Running Tests

```bash
cd e2e
go run ./cmd/e2e --profile ml-model-selection --test model-selection
```

## E2E Model Loading Flow

The E2E test automatically handles model loading:

```
┌─────────────────────────────────────────────────────────┐
│              Automatic Model Loading                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. Check if models exist locally                       │
│     └─ YES → Use local models ✓                         │
│     └─ NO  → Continue...                                │
│                                                         │
│  2. Install huggingface-hub (if needed)                 │
│     └─ pip install huggingface-hub                      │
│                                                         │
│  3. Download from HuggingFace                           │
│     (abdallah1008/semantic-router-ml-models)            │
│     └─ SUCCESS → Use downloaded models ✓                │
│     └─ FAIL    → Error                                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Troubleshooting

### "No decision matched" - 0% Accuracy

```
Signal evaluation results: domain=[health]
"No decision matched"
selected=mistral-7b  (fallback)
```

**Cause:** Domain name mismatch between classifier output and decision rules.

**Fix:** Ensure decision rules use exact VSR category names (with spaces):

```yaml
# ❌ WRONG - uses underscore
conditions:
  - type: "domain"
    name: "computer science"

# ✅ CORRECT - uses space
conditions:
  - type: "domain"
    name: "computer science"
```

### Models Not Found

```
Error: pretrained model not found
```

The E2E test automatically downloads models from HuggingFace. If download fails:

1. Check internet connectivity
2. Verify the HuggingFace repository is accessible: https://huggingface.co/abdallah1008/semantic-router-ml-models
3. Try manual download:

```bash
pip install huggingface-hub
cd src/training/model_selection/ml_model_selection
python download_model.py --output-dir ../../../.cache/ml-models
```

### No Model Selected

If fallback model is always selected:

1. Check decision conditions match query domain **exactly** (spaces, not underscores)
2. Verify pretrained models are loaded (check logs)
3. Ensure embedding model is available
4. Check that your decision covers the detected domain (see logs for `domain=[...]`)

### Low Accuracy

If selection accuracy is low:

1. Retrain with more benchmark data
2. Adjust K value for KNN (try k=3 or k=7)
3. For SVM, try gamma=0.5 or gamma=2.0
4. Ensure all 14 VSR categories have corresponding decisions
