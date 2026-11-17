---
title: Train and Publish Router Embedding Models
sidebar_label: Router Embeddings & Hugging Face
---

# Train and Publish Router Embedding Models

vLLM Semantic Router uses dense embedding models for recall only: they find
the candidates a feature then uses or judges, and no routing decision reads
them. Routing questions go to the decision model; see
[System One](../system-one/overview). Recall is where a domain-adapted
embedding model pays off:

1. **Semantic Caching**: recalling the cached requests that may be safe paraphrases of an incoming query. The `same_request` judge then confirms each hit, so a recalled request with a different meaning still gets a fresh generation.
2. **Retrieval**: recalling the documents, memories and vector-store entries a request needs, before a `relevance` judge keeps the useful ones.

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
Semantic Router Deployment (Recall for the Cache, Memory or Retrieval)
```

---

## Pipeline A: Semantic-Cache LoRA Adapters

The cache embedding pipeline fine-tunes parameter-efficient Low-Rank Adaptation (LoRA) adapters on an encoder base model. The goal is sharpening the boundary between genuine paraphrases (safe to reuse cached outputs) and topically related queries (which must miss the cache).

Code location: [`src/training/model_embeddings/cache_embeddings/`](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_embeddings/cache_embeddings)

### 1. Triplet Data Format

Training consumes JSON Lines (`.jsonl`) files where each row specifies a contrastive triplet:

```json
{"anchor": "Which diagnostic tests confirm type 2 diabetes?", "positive": "How is type 2 diabetes diagnosed?", "negative": "How is type 2 diabetes managed with metformin?"}
```

- **Anchor**: The generated paraphrase variant (the query tested against the cache).
- **Positive**: The original seed query sharing identical semantic intent (safe to reuse).
- **Negative**: A query from the same domain that shares terminology but represents a different factual question (must miss the cache).

### 2. Triplet Generation

Use `generate_training_data.py` with domain prompt definitions from `domains/prompts.yaml`. The script does not query an external HTTP endpoint; instead, it imports `vllm.LLM` directly in-process, downloading and loading the generation model onto the local GPU(s) using the configured `--tensor-parallel` settings.

Input files must be JSON Lines where each row provides a seed query in the format `{"query": "..."}`:

```bash
python src/training/model_embeddings/cache_embeddings/generate_training_data.py \
  --input data/medical/unlabeled_queries.jsonl \
  --domain medical \
  --output data/medical/triplets.jsonl \
  --max-queries 5000
```

Always perform a manual label review on synthetic triplet batches to prune hallucinated negatives or near-duplicate positives before training.

### 3. LoRA Adapter Fine-Tuning

Train the LoRA adapter with Multiple Negatives Ranking (MNR) loss using `--lr`:

```bash
python src/training/model_embeddings/cache_embeddings/lora_trainer.py \
  --train-data data/medical/triplets.jsonl \
  --val-data data/medical/val_triplets.jsonl \
  --base-model sentence-transformers/all-MiniLM-L12-v2 \
  --output models/medical-cache-lora \
  --epochs 2 \
  --lr 2e-4
```

To create a **multi-domain adapter**, concatenate reviewed domain datasets (e.g. medical, legal, code) and train in a single pass.

### 4. Evaluation & Similarity Margin

Evaluate separation quality using `evaluate_multi_domain.py`:

```bash
python src/training/model_embeddings/cache_embeddings/evaluate_multi_domain.py \
  --lora-path models/medical-cache-lora \
  --sample-size 2000
```

By default, the evaluator looks for domain test sets locally:

- `--medical-test data/cache_embeddings/medical/test_set.jsonl`
- `--law-test data/cache_embeddings/law/test_triplets.jsonl`
- `--programming-test data/cache_embeddings/programming/test_triplets.jsonl`

When these local files are absent, the evaluator downloads them from `vllm-sr/cache-embedding-test-sets` on Hugging Face Hub. Because this repository requires authenticated access, unauthenticated requests will return HTTP 401. Ensure you either supply your own local test triplet files via the flags above, or set an authorized token via `export HF_TOKEN="<token>"`.

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

To deploy the resulting checkpoint directly inside Semantic Router's built-in `model_runtime`, fine-tune a supported ModernBERT (`mmbert`) or Qwen3 embedding backbone (such as `vllm-sr/mmbert-embed-32k-2d-matryoshka`). Classic BERT or MiniLM backbones (`sentence-transformers/all-MiniLM-L12-v2`) can also be trained with this script, but must be served behind an external OpenAI-compatible embedding endpoint because `model_runtime` does not include a classic BERT/MiniLM embedding family.

```bash
python src/training/model_embeddings/domain_adapted_embeddings/train.py \
  --data-dir data/processed \
  --output-dir models/domain-adapted-fin \
  --base-model vllm-sr/mmbert-embed-32k-2d-matryoshka \
  --iterations 2 \
  --learning-rate 5e-5 \
  --margin 0.1
```

*Reference:* The mining methodology is based on [“Adaptation of Embedding Models to Financial Filings via LLM Distillation”](https://arxiv.org/abs/2512.08088).

### 3. Retrieval Metrics

Evaluate Mean Reciprocal Rank (MRR@5) and Recall on the held-out test split. On the MedQuAD medical Q&A benchmark, domain adaptation achieved a +71.18% MRR@5 improvement over the baseline (evaluated in `src/training/model_embeddings/domain_adapted_embeddings/train.py` with `lr=5e-5`). Results from one domain do not establish performance across other domains; always evaluate on representative held-out domain splits.

---

## Packaging and Model Provenance

Before publication, verify that the checkpoint directory contains the required artifacts for its packaging type.

### Domain-Adapted Full Embedding Checkpoints

1. **Model Weights**: `model.safetensors`.
2. **Model & Sentence-Transformer Configuration**: `config.json`, `modules.json`, and pooling configuration (e.g. `1_Pooling/config.json`).
3. **Tokenizer**: `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, and `vocab.txt` (or `sentencepiece.bpe.model`).
4. **Model Card (`README.md`)**: Base model lineage, training datasets/splits/seeds, evaluated metrics (MRR@5, recall), intended use, and license.

### Semantic-Cache LoRA Adapter Checkpoints

LoRA training calls PEFT `save_pretrained`, which emits adapter weights and adapter metadata rather than a standalone base-model `config.json`:

1. **Adapter Weights**: `adapter_model.safetensors`.
2. **Adapter Configuration**: `adapter_config.json` (recording `base_model_name_or_path` and target LoRA modules).
3. **Tokenizer (optional)**: `tokenizer.json` and `tokenizer_config.json` when saved alongside the adapter.
4. **Model Card (`README.md`)**: Base model identifier, domain training splits, evaluated similarity margin, and license.

---

## Publishing to Hugging Face Hub

### Safe Authentication

Never commit API tokens or hardcode credentials in code or repository scripts. Use the standard `HF_TOKEN` environment variable:

```bash
export HF_TOKEN="hf_your_write_token"
huggingface-cli login --token "$HF_TOKEN"
```

### Uploading Artifacts

Upload models or datasets to Hugging Face Hub using your personal or organization namespace:

```bash
# Upload a trained LoRA adapter
cd models/medical-cache-lora
huggingface-cli upload <your-hf-namespace>/medical-cache-lora . --repo-type model

# Upload a domain-adapted full embedding model (mmBERT backbone)
cd models/domain-adapted-fin/best
huggingface-cli upload <your-hf-namespace>/mmbert-embed-financial . --repo-type model
```

*Note:* Official models maintained by the project live under the [`vllm-sr`](https://huggingface.co/vllm-sr) organization.

---

## Using Published Models in Semantic Router

### 1. Router Deployment & Model Catalog Binding

Loading a LoRA adapter directly in the router is a separate runtime capability; a cache LoRA can currently be tested offline through Python and PEFT, but cannot be activated by Router configuration.

To deploy a domain-adapted full embedding model built on a supported ModernBERT (`mmbert`) or Qwen3 backbone inside Semantic Router's local `model_runtime`, declare it as a deployment and name it in `global.model_catalog.embeddings.default`, the deployment every recall feature embeds with:

```yaml
version: v0.5

listeners:
  - name: default
    address: 0.0.0.0
    port: 8080

global:
  model_catalog:
    deployments:
      domain-embedding:
        provider: model_runtime
        artifact: your-org/mmbert-embed-financial
        device: cpu
        input:
          max_tokens: 8192
          overflow: truncate
    embeddings:
      default: {deployment: domain-embedding}
```

Replace `your-org` with the namespace you published to, and set the
deployment's `revision` to the commit you evaluated so the Router serves
exactly that model.

If you fine-tuned a classic BERT/MiniLM checkpoint (such as a 384-dimensional `all-MiniLM-L12-v2` derivative) or prefer serving embeddings via an external OpenAI-compatible HTTP server, declare that server as an `openai_embeddings` deployment, with the width and input of its vectors in `capabilities`:

```yaml
version: v0.5

listeners:
  - name: default
    address: 0.0.0.0
    port: 8080

global:
  model_catalog:
    deployments:
      financial-minilm:
        provider: openai_embeddings
        endpoint: http://model-service:8000/v1
        model: your-org/financial-minilm-l12
        capabilities: {dimensions: 384, modalities: [text]}
    embeddings:
      default: {deployment: financial-minilm}
```

A store indexed with another model keeps its own `embedding: {deployment, dimension}` block, so only stores without one move to the new model. Vectors of different models are never compared: re-index a store after you change the model it embeds with (see [Embeddings](../model-runtime/guides/embeddings)).

### 2. Direct Python & PEFT Inference

To verify LoRA adapter inference in local evaluation scripts using `sentence-transformers` and `peft`:

```python
from sentence_transformers import SentenceTransformer
from peft import PeftModel

# Load base model
base = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

# Attach the published LoRA adapter
base[0].auto_model = PeftModel.from_pretrained(
    base[0].auto_model,
    "<your-hf-namespace>/medical-cache-lora"
)

# Generate domain-optimized embeddings
query = "What clinical indications support a metformin prescription?"
embedding = base.encode(query, normalize_embeddings=True)
```
