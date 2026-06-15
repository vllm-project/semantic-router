# Domain-Adapted Embedding Fine-Tuning

![Domain-Adapted Embeddings](domain-embed.jpg)

Fine-tune embedding models for improved retrieval in specific domains using iterative hard-negative mining.

**Based on:** ["Distilling an LLM's Wisdom: A Framework for Creating Domain Adapted Financial Embedding Models"](https://arxiv.org/abs/2512.08088)

---

## Pre-trained Models

| Domain | Model | Base Model | Improvement |
|--------|-------|------------|-------------|
| Medical | [mmbert-embed-medical](https://huggingface.co/llm-semantic-router/mmbert-embed-medical) | mmbert-embed-32k-2d-matryoshka | +71% MRR@5 |

---

## The Problem

General-purpose embedding models work well across domains but underperform on specialized content (medical, legal, financial). When used in a semantic router, this means less accurate routing for domain-specific queries.

## The Solution

**Iterative hard-negative mining** fine-tunes embeddings on domain data:

1. **Mine triplets** from current model's rankings using ground-truth labels
   - Hard triplets: Ground-truth docs ranked low + non-GT docs ranked high
   - Easy triplets: Ground-truth docs already ranked high (prevents forgetting)

2. **Fine-tune** with TripletLoss (margin=0.1)

3. **Repeat** for 2 iterations (accumulating triplets)

> **Note:** The original paper uses an LLM-as-judge to score relevance. Our implementation
> uses ground-truth labels instead (which doc answers which query), achieving similar results
> without requiring an LLM server.

---

## Results

### MedQuAD Medical Dataset (13,125 training queries)

| Metric | Baseline | Iteration 1 | Iteration 2 | Improvement |
|--------|----------|-------------|-------------|-------------|
| **MRR@5** | 0.4354 | 0.6528 (+49.93%) | **0.7453** | **+71.18%** |
| **Recall@5** | 0.4896 | 0.5918 | **0.6900** | **+40.92%** |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | llm-semantic-router/mmbert-embed-32k-2d-matryoshka |
| Learning rate | 5e-5 |
| Batch size | 8 |
| Epochs per iteration | 2 |
| Margin | 0.1 |
| Easy:Hard ratio | 2:1 |
| GPU | NVIDIA L4 (24GB) |

---

## Limitation: Multi-Domain Integration (TBD)

Each domain-adapted model is fine-tuned independently. When multiple domains exist (medical, legal, finance, etc.), a strategy is needed to integrate them into the semantic router:

| Option | Approach | Trade-off |
|--------|----------|-----------|
| **Multi-domain fine-tuning** | Train one model on all domains together | Simple deployment, but domains may interfere |
| **Two-stage routing** | Domain classifier → domain-specific embeddings | Best accuracy, more complexity |
| **Ensemble** | Run through multiple models, combine scores | Handles cross-domain queries, higher latency |

Currently, this pipeline produces standalone domain models. Integration with the semantic router's existing domain signal infrastructure is future work.

---

## Usage

See [USAGE.md](USAGE.md) for installation, training instructions, and code examples.

---

## References

- **Paper:** [arXiv:2512.08088](https://arxiv.org/abs/2512.08088) - "Distilling an LLM's Wisdom"
- **Base Model:** [llm-semantic-router/mmbert-embed-32k-2d-matryoshka](https://huggingface.co/llm-semantic-router/mmbert-embed-32k-2d-matryoshka)
- **Dataset:** [MedQuAD](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset)
