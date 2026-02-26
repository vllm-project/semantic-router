# Cache Embedding LoRA Training

Training domain-specific and multi-domain LoRA adapters for semantic cache embeddings.

## Overview

This directory contains scripts for training LoRA (Low-Rank Adaptation) adapters that improve semantic cache accuracy for domain-specific queries. LoRA adapters are small (≈600KB) fine-tuned models that enhance the base embedding model's ability to distinguish between semantically similar but different queries.

## Pre-trained Models

All trained models are available on HuggingFace:

### Recommended: Multi-Domain LoRA

- **[multi-domain-cache-lora-L12](https://huggingface.co/llm-semantic-router/multi-domain-cache-lora-L12)** - Single adapter for all domains
  - Medical: +24.9% improvement
  - Law: +27.3% improvement
  - Programming: +12.4% improvement
  - **Use this for production** ✅

### Domain-Specific LoRAs

- **[medical-cache-lora](https://huggingface.co/llm-semantic-router/medical-cache-lora)** - Medical/health queries only (+42.8%)
- **[law-cache-lora](https://huggingface.co/llm-semantic-router/law-cache-lora)** - Legal queries only (+25.9%)
- **[programming-cache-lora](https://huggingface.co/llm-semantic-router/programming-cache-lora)** - NOT recommended (+0.4%, use multi-domain instead)
- **[psychology-cache-lora](https://huggingface.co/llm-semantic-router/psychology-cache-lora)** - Psychology queries only (no test set yet)

## Test Datasets

Test datasets are available on HuggingFace:

- **[cache-embedding-test-sets](https://huggingface.co/datasets/llm-semantic-router/cache-embedding-test-sets)**
  - Medical: 200 triplets
  - Law: 20,862 triplets
  - Programming: 20,862 triplets

## Quick Start

### 1. Generate Training Data

```bash
python3 src/training/cache_embeddings/generate_training_data.py \
  --domain medical \
  --output data/cache_embeddings/medical/triplets.jsonl \
  --num-samples 10000
```

### 2. Train Domain-Specific LoRA

```bash
python3 src/training/cache_embeddings/lora_trainer.py \
  --train-data data/cache_embeddings/medical/triplets.jsonl \
  --base-model sentence-transformers/all-MiniLM-L12-v2 \
  --output models/cache/medical-cache-lora \
  --epochs 1
```

### 3. Train Multi-Domain LoRA

```bash
# Combine all domain triplets
cat data/cache_embeddings/medical/triplets.jsonl \
    data/cache_embeddings/law/triplets.jsonl \
    data/cache_embeddings/programming/triplets.jsonl \
    data/cache_embeddings/psychology/triplets.jsonl \
    > data/cache_embeddings/multi-domain/triplets.jsonl

# Train on combined data
python3 src/training/cache_embeddings/lora_trainer.py \
  --train-data data/cache_embeddings/multi-domain/triplets.jsonl \
  --base-model sentence-transformers/all-MiniLM-L12-v2 \
  --output models/multi-domain-cache-lora-L12 \
  --epochs 1
```

### 4. Evaluate Model

```bash
python3 src/training/cache_embeddings/evaluate_multi_domain.py \
  --base-model sentence-transformers/all-MiniLM-L12-v2 \
  --lora-model models/multi-domain-cache-lora-L12 \
  --test-file data/cache_embeddings/medical/test_set.jsonl \
  --output results.json
```

## Using Pre-trained Models

### Download from HuggingFace

```python
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from peft import PeftModel

# Load base model
base_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

# Download and apply LoRA adapter
base_model[0].auto_model = PeftModel.from_pretrained(
    base_model[0].auto_model,
    "llm-semantic-router/multi-domain-cache-lora-L12"
)

# Use for embeddings
embedding = base_model.encode("What are the symptoms of diabetes?")
```

### Download Test Dataset

```python
from huggingface_hub import hf_hub_download
import json

# Download test set
test_file = hf_hub_download(
    repo_id="llm-semantic-router/cache-embedding-test-sets",
    filename="medical/test_set.jsonl",
    repo_type="dataset"
)

# Load triplets
triplets = []
with open(test_file) as f:
    for line in f:
        triplets.append(json.loads(line))
```

## Configuration

### Config File: `config/config.yaml`

**Note:** LoRA integration requires code changes to support loading adapters. The configuration below shows the intended usage once implemented.

```yaml
semantic_cache:
  enabled: true
  backend_type: "memory"
  similarity_threshold: 0.8

  # LoRA model (requires base model: sentence-transformers/all-MiniLM-L12-v2)
  # This feature requires implementation of LoRA loading in the semantic cache module
  lora_model: "llm-semantic-router/multi-domain-cache-lora-L12"
```

## Training on AWS

See [aws/README.md](aws/README.md) for instructions on launching GPU instances and training at scale.

### Quick AWS Launch

```bash
# Launch g5.12xlarge instance (4x A10G GPUs)
src/training/cache_embeddings/aws/deploy-vllm.sh
```

## Scripts

| Script | Purpose |
|--------|---------|
| `generate_training_data.py` | Generate synthetic training triplets |
| `lora_trainer.py` | Train LoRA adapter on triplets |
| `evaluate_multi_domain.py` | Evaluate LoRA on test set |
| `test_lora_model.py` | Quick test of trained model |
| `train-domain.sh` | Helper script for training |

## Domain Configuration

Domain-specific prompts and settings are in `domains/prompts.yaml`.

## Performance Metrics

### Margin Score

The primary metric for cache quality:

```text
margin = avg(positive_similarity) - avg(negative_similarity)
```

**Higher margin = Better cache accuracy**

### Baseline Margins (No LoRA)

- Medical: 0.44
- Law: 0.49
- Programming: 0.24 (harder task - negatives are very similar)

### With Multi-Domain LoRA

- Medical: 0.55 (+24.9%)
- Law: 0.63 (+27.3%)
- Programming: 0.27 (+12.4%)

## Architecture

### Base Model

- **sentence-transformers/all-MiniLM-L12-v2** (90MB)
- 12-layer transformer
- 384-dimensional embeddings
- Fast inference on CPU

### LoRA Adapter

- ≈600KB per adapter
- Rank: 8
- Alpha: 16
- Target modules: query, value projections
- Training: 1 epoch on triplet loss

### Memory Usage

- Base model: 90 MB
- Single LoRA: 0.6 MB
- **Total: 90.6 MB** (multi-domain)

## Production Recommendations

### Use Multi-Domain LoRA

**Why:**

- ✅ Works well across all domains
- ✅ No domain detection needed
- ✅ Simple deployment
- ✅ Small memory overhead
- ✅ Better generalization

**Configuration:**

```yaml
semantic_cache:
  lora_model: "llm-semantic-router/multi-domain-cache-lora-L12"
```

### Alternative: Hybrid Approach

For maximum medical accuracy:

1. Detect medical queries
2. Use `medical-cache-lora` for medical (+42.8%)
3. Use `multi-domain-cache-lora-L12` for others

Requires domain detection classifier.

## Triplet Data Format

Training data should be JSONL with triplets:

```json
{
  "anchor": "What are the symptoms of acute myocardial infarction?",
  "positive": "How do patients present with acute myocardial infarction?",
  "negative": "What is the mechanism of action for beta-blockers?"
}
```

**Key properties:**

- **Anchor**: Original query
- **Positive**: Semantically equivalent (should cache)
- **Negative**: Related but different (should NOT cache)

## Citation

If you use these models or training scripts, please cite:

```bibtex
@misc{semantic-router-cache-lora,
  title={LoRA Adapters for Semantic Cache Embeddings},
  author={vLLM Project},
  year={2025},
  publisher={HuggingFace},
  howpublished={\url{https://huggingface.co/llm-semantic-router}}
}
```

## License

Apache 2.0

## Contact

For questions or issues, open an issue on [vllm-project/semantic-router](https://github.com/vllm-project/semantic-router).
