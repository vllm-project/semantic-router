---
title: Train and Publish Router Embedding Models
sidebar_label: Router Embeddings & Hugging Face
---

# Train and Publish Router Embedding Models

vLLM Semantic Router uses dense embedding models for two core inference capabilities:

1. **Semantic Routing**: Mapping user prompts to topic anchors and intent categories using vector similarity signals.
2. **Semantic Caching**: Determining whether an incoming query is a safe paraphrase of a previously cached response or a distinct query requiring fresh generation.

While general-purpose embedding models (such as `sentence-transformers/all-MiniLM-L12-v2` or `vllm-sr/Vela-1.0-Encoder-307M`) perform well on standard language benchmarks, specialized technical and enterprise domains (e.g. medical, legal, financial, programming) often require domain-specific fine-tuning.

This guide details the two maintained embedding training pipelines in [`src/training/model_embeddings/`](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_embeddings), evaluation workflows, safe artifact packaging, Hugging Face Hub publication, and configuration inside Semantic Router.

---

## End-to-End Workflow

```text
Domain Data (Queries / Q&A)
        │
        ▼
Triplet Construction & Negative Mining (Anchor, Positive, Negative)
        │
        ▼
Fine-Tuning (LoRA Adapter or Domain-Adapted Full Model)
        │
        ▼
Evaluation (Separation Margin / MRR@5 / Recall)
        │
        ▼
Packaging & Hub Upload (huggingface-cli with HF_TOKEN)
        │
        ▼
Semantic Router Deployment (Semantic Cache or Routing Signals)
```

---

## Pipeline A: Semantic-Cache LoRA Adapters

The cache embedding pipeline fine-tunes parameter-efficient Low-Rank Adaptation (LoRA) adapters on an encoder base model. The goal is sharpening the boundary between genuine paraphrases (safe to reuse cached outputs) and topically related queries (which must miss the cache).

Code location: [`src/training/model_embeddings/cache_embeddings/`](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_embeddings/cache_embeddings)

### 1. Triplet Data Format

Training consumes JSON Lines (`.jsonl`) files where each row specifies a contrastive triplet:

```json
{"anchor": "How is type 2 diabetes diagnosed?", "positive": "Which diagnostic tests confirm type 2 diabetes?", "negative": "How is type 2 diabetes managed with metformin?"}
```

- **Anchor**: The reference query.
- **Positive**: A true paraphrase with identical semantic intent.
- **Negative**: A query from the same domain that shares terminology but represents a different factual question.

### 2. Triplet Generation

Use `generate_training_data.py` with domain prompt definitions from `domains/prompts.yaml` and a local vLLM generation endpoint:

```bash
python src/training/model_embeddings/cache_embeddings/generate_training_data.py \
  --input data/medical/unlabeled_queries.jsonl \
  --domain medical \
  --output data/medical/triplets.jsonl \
  --max-queries 5000
```

Always perform a manual label review on synthetic triplet batches to prune hallucinated negatives or near-duplicate positives before training.

### 3. LoRA Adapter Fine-Tuning

Train the LoRA adapter with Multiple Negatives Ranking (MNR) loss:

```bash
python src/training/model_embeddings/cache_embeddings/lora_trainer.py \
  --train-data data/medical/triplets.jsonl \
  --val-data data/medical/val_triplets.jsonl \
  --base-model sentence-transformers/all-MiniLM-L12-v2 \
  --output models/medical-cache-lora \
  --epochs 2 \
  --learning-rate 2e-4
```

To create a **multi-domain adapter**, concatenate reviewed domain datasets (e.g. medical, legal, code) and train in a single pass.

### 4. Evaluation & Similarity Margin

Evaluate separation quality using `evaluate_multi_domain.py`:

```bash
python src/training/model_embeddings/cache_embeddings/evaluate_multi_domain.py \
  --lora-path models/medical-cache-lora \
  --sample-size 2000
```

The primary metric is the **mean similarity margin**:

$$\text{Margin} = \frac{1}{N} \sum_{i=1}^N \text{sim}(\text{anchor}_i, \text{positive}_i) - \frac{1}{N} \sum_{i=1}^N \text{sim}(\text{anchor}_i, \text{negative}_i)$$

A larger margin demonstrates that paraphrases cluster tightly while non-paraphrases are pushed outside the decision threshold.

---

## Pipeline B: Domain-Adapted Full Embeddings

When domain terminology requires dense representation adaptation across the entire vocabulary, fine-tune the full embedding model with iterative hard-negative mining.

Code location: [`src/training/model_embeddings/domain_adapted_embeddings/`](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_embeddings/domain_adapted_embeddings)

### 1. Data Preparation

Format Q&A pairs where each question links to its ground-truth answering passage:

```bash
python src/training/model_embeddings/domain_adapted_embeddings/prepare_data.py \
  --source jsonl \
  --input-file data/financial_qa.jsonl \
  --output-dir data/processed
```

### 2. Iterative Hard-Negative Mining & Triplet Loss

The training workflow:

1. Encodes corpus passages and queries using the current model checkpoint.
2. Mines hard negatives (irrelevant passages ranked undesirably high) and hard positives (true passages ranked low).
3. Blends hard triplets with easy triplets (ratio ~ 2:1) to prevent catastrophic forgetting.
4. Optimizes Triplet Margin Loss (margin $\approx 0.1$) with a learning rate of $5\times 10^{-5}$.
5. Accumulates mined triplets across iterations.

```bash
python src/training/model_embeddings/domain_adapted_embeddings/train.py \
  --data-dir data/processed \
  --output-dir models/domain-adapted-fin \
  --base-model sentence-transformers/all-MiniLM-L12-v2 \
  --iterations 2 \
  --learning-rate 5e-5 \
  --margin 0.1
```

*Reference:* The mining methodology is based on [“Adaptation of Embedding Models to Financial Filings via LLM Distillation”](https://arxiv.org/abs/2512.08088).

### 3. Retrieval Metrics

Evaluate Mean Reciprocal Rank (MRR@5) and Recall on the held-out test split. Domain adaptation routinely achieves +50% to +70% MRR@5 improvements over untuned general-purpose baselines on dense technical texts.

---

## Packaging and Model Provenance

Before publication, ensure the checkpoint directory contains all required artifacts:

1. **Model Weights**: `model.safetensors` or `adapter_model.safetensors`.
2. **Tokenizer**: `tokenizer.json`, `tokenizer_config.json`, `vocab.txt` (or `sentencepiece.bpe.model`).
3. **Configuration**: `config.json` and (for LoRA) `adapter_config.json`.
4. **Model Card (`README.md`)**:
   - Explicit base model lineage.
   - Training datasets, splits, and seeds.
   - Evaluated metrics (margin, MRR@5) alongside baseline comparisons.
   - Intended use and domain limitations.
   - License (e.g. Apache-2.0 or MIT).

---

## Publishing to Hugging Face Hub

### Safe Authentication

Never commit API tokens or hardcode credentials in code or repository scripts. Use the standard `HF_TOKEN` environment variable:

```bash
export HF_TOKEN="hf_your_write_token"
huggingface-cli login --token "$HF_TOKEN"
```

### Uploading Artifacts

Upload models or datasets to the community organization:

```bash
# Upload a trained LoRA adapter
cd models/medical-cache-lora
huggingface-cli upload llm-semantic-router/medical-cache-lora . --repo-type model

# Upload a domain-adapted full embedding model
cd models/domain-adapted-fin/best
huggingface-cli upload llm-semantic-router/financial-embedding-l12 . --repo-type model
```

---

## Using Published Models in Semantic Router

### 1. Semantic Cache Configuration

Reference the published model in your Semantic Router configuration file:

```yaml
version: v0.3

providers:
  - name: embedding-backend
    type: vllm
    endpoint: http://model-service:8000/v1
    model: sentence-transformers/all-MiniLM-L12-v2

listeners:
  - name: llm-entrypoint
    address: 0.0.0.0:8080
    plugins:
      - name: semantic-cache
        configuration:
          similarity_threshold: 0.88
          embedding_provider: embedding-backend
          lora_adapter: llm-semantic-router/medical-cache-lora
```

### 2. Direct Python Inference

To verify adapter inference in local scripts using `sentence-transformers` and `peft`:

```python
from sentence_transformers import SentenceTransformer
from peft import PeftModel

# Load base model
base = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

# Attach the published LoRA adapter
base[0].auto_model = PeftModel.from_pretrained(
    base[0].auto_model,
    "llm-semantic-router/medical-cache-lora"
)

# Generate domain-optimized embeddings
query = "What clinical indications support a metformin prescription?"
embedding = base.encode(query, normalize_embeddings=True)
```
