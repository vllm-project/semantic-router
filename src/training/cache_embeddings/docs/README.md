# Cache Embedding Training Pipeline

**Production-grade pipeline for training domain-specific cache embedding models using LLM-generated synthetic data.**

Based on: [arXiv:2504.02268v1](https://arxiv.org/pdf/2504.02268v1)

## Overview

This pipeline fine-tunes lightweight LoRA adapters for semantic caching using LLM-generated training data. A single multi-domain LoRA adapter achieves **+19.4% average improvement** across 4 validated domains (Medical, Law, Programming, Psychology) while adding only 0.6MB to the base model.

### Multi-Domain Results

| Domain | Improvement | Triplets | Notes |
|--------|-------------|----------|-------|
| **Medical** | **+14.6%** | 112,737 | Healthcare, clinical terms |
| **Law** | **+16.9%** | 127,739 | Case law, legal concepts |
| **Programming** | **+11.3%** | 208,620 | Code, technical documentation |
| **Psychology** | **+34.9%** | ~169,000 | Mental health, theories |
| **Average** | **+19.4%** | ~618,096 total | Single 582KB adapter |

**Key Benefits:**
- ✅ One adapter works across all domains
- ✅ Memory: Base (584MB) + adapter (0.6MB) = 585MB total
- ✅ No domain switching logic needed
- ✅ Simpler deployment than per-domain models

**Training Data:**
- Pre-generated triplets available at: https://huggingface.co/datasets/llm-semantic-router/semantic-router-cache-triplets (private)
- Includes all 4 domains + merged multi-domain triplets (~618K total)

## Quick Start

### Prerequisites

**For production (GPU with vLLM):**
- AWS EC2 with NVIDIA GPUs (e.g., g5.12xlarge with 4x A10G)
- Python 3.11+ with: `vllm`, `transformers`, `torch`, `peft`, `sentence-transformers`

**For local testing (CPU with Ollama):**
- Ollama installed with a model (e.g., `qwen2.5:1.5b`)
- Python 3.11+ with: `transformers`, `torch`, `peft`, `sentence-transformers`

### Single-Domain Training (Two-Step Pipeline)

**Step 1: Generate training triplets** (~1.5-2 hours on 4x A10G for 44K queries)

```bash
python3 generate_training_data.py \
  --input data/cache_embeddings/medical/unlabeled_queries.jsonl \
  --output data/cache_embeddings/medical/triplets.jsonl \
  --backend vllm \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --paraphrases 3 \
  --negatives 2 \
  --batch-size 32 \
  --tensor-parallel 4
```

**Step 2: Train LoRA adapter** (~5 minutes on GPU for 130K triplets)

```bash
python3 lora_trainer.py \
  --train-data data/cache_embeddings/medical/triplets.jsonl \
  --base-model sentence-transformers/all-MiniLM-L12-v2 \
  --output models/medical-cache-lora \
  --epochs 1 \
  --batch-size 32 \
  --lr 2e-5 \
  --temperature 0.05
```

**Result:** 582KB LoRA adapter ready for deployment.

For AWS deployment automation, see [QUICK_START_AWS.md](QUICK_START_AWS.md).

### Multi-Domain Training (Recommended)

**Important Concept**: You merge **triplets** (training data), not adapters. Each time you add a new domain, you retrain from scratch on the combined triplet dataset.

#### The Multi-Domain Flow

```
Domain A queries → Generate triplets → medical_triplets.jsonl
Domain B queries → Generate triplets → law_triplets.jsonl
Domain C queries → Generate triplets → programming_triplets.jsonl
                                          ↓
                        Merge all triplets using `cat`
                                          ↓
                        multi-domain_triplets.jsonl
                                          ↓
                    Train LoRA adapter from scratch
                                          ↓
                    multi-domain-lora-adapter (582KB)
                    Works on ALL domains A+B+C
```

**Key Point**: When adding domain D later, merge A+B+C+D triplets and retrain. You don't "add to" the existing adapter.

#### Commands

```bash
# Step 1: Generate triplets for each domain (run separately)
python3 generate_training_data.py \
  --input data/cache_embeddings/medical/unlabeled_queries.jsonl \
  --output data/cache_embeddings/medical/triplets.jsonl \
  ...

python3 generate_training_data.py \
  --input data/cache_embeddings/law/unlabeled_queries.jsonl \
  --output data/cache_embeddings/law/triplets.jsonl \
  ...

# Step 2: Merge triplets from all domains
cat data/cache_embeddings/medical/triplets.jsonl \
    data/cache_embeddings/law/triplets.jsonl \
    data/cache_embeddings/programming/triplets.jsonl \
    data/cache_embeddings/psychology/triplets.jsonl \
  > data/cache_embeddings/multi-domain/triplets.jsonl

# Step 3: Train single multi-domain adapter from scratch
python3 lora_trainer.py \
  --train-data data/cache_embeddings/multi-domain/triplets.jsonl \
  --base-model sentence-transformers/all-MiniLM-L12-v2 \
  --output models/multi-domain-cache-lora \
  --epochs 1 \
  --batch-size 32 \
  --lr 2e-5
```

**Result:** Single 582KB adapter that works across all domains with **+19.4% average improvement**.

#### Adding New Domains Later

```bash
# You have: medical + law + programming + psychology (618K triplets)
# You want to add: biology (50K triplets)

# 1. Generate biology triplets
python3 generate_training_data.py \
  --input data/cache_embeddings/biology/unlabeled_queries.jsonl \
  --output data/cache_embeddings/biology/triplets.jsonl

# 2. Merge with ALL existing triplets
cat data/cache_embeddings/medical/triplets.jsonl \
    data/cache_embeddings/law/triplets.jsonl \
    data/cache_embeddings/programming/triplets.jsonl \
    data/cache_embeddings/psychology/triplets.jsonl \
    data/cache_embeddings/biology/triplets.jsonl \
  > data/cache_embeddings/multi-domain-v2/triplets.jsonl

# 3. Retrain from scratch on merged dataset (668K triplets now)
python3 lora_trainer.py \
  --train-data data/cache_embeddings/multi-domain-v2/triplets.jsonl \
  --base-model sentence-transformers/all-MiniLM-L12-v2 \
  --output models/multi-domain-cache-lora-v2 \
  --epochs 1
```

This ensures the new adapter maintains performance on all previous domains while adding biology.

## How It Works

### 1. Triplet-Based Training Data

Each training sample contains three components:

```json
{
  "anchor": "What are the diagnostic methods for whooping cough?",
  "positive": "How to diagnose Pertussis?",
  "negative": "What are the symptoms and signs of Pertussis?",
  "is_duplicate": 1
}
```

- **Anchor** ↔ **Positive**: Semantically identical (paraphrase) → HIGH similarity
- **Anchor** ↔ **Negative**: Related but different intent → LOW similarity

### 2. LLM-Generated Synthetic Data

Starting with unlabeled domain queries, we use an LLM (Qwen 2.5 1.5B) to generate:
- **3 paraphrases** per query (positives that preserve meaning)
- **2 hard negatives** per query (related but different intent)

For 44K queries: 44K × 3 paraphrases = 132K anchor-positive pairs → ~130K complete triplets

### 3. LoRA Fine-Tuning

**Base Model:** `sentence-transformers/all-MiniLM-L12-v2` (33.5M params, 384-dim embeddings)

**LoRA Configuration:**
- Rank: 8, Alpha: 32
- Target modules: query, value projection layers
- Trainable params: 147K (0.44% of base model)
- Output: 582KB adapter file

**Training:**
- Loss: Multiple Negatives Ranking (MNR) with temperature=0.05
- Epochs: 1 (sufficient per paper)
- Batch size: 32
- Learning rate: 2e-5

## Files

| File | Purpose |
|------|---------|
| `generate_training_data.py` | **Step 1:** Creates triplets from unlabeled queries using vLLM or Ollama |
| `lora_trainer.py` | **Step 2:** Trains LoRA adapter with MNR loss (includes loss implementation) |
| `evaluate_multi_domain.py` | Validates multi-domain LoRA performance across domains |
| `test_lora_model.py` | Evaluates single-domain LoRA vs baseline |
| `common_utils.py` | Shared utilities (logging, data I/O, seeding) |

## Datasets

Links to datasets used for training (not included in repo):

- **Medical:** [MedQuAD](https://github.com/abachaa/MedQuAD) (CC BY 4.0) - 44,603 medical queries from NIH, CDC, Mayo Clinic
- **Law:** [lex_glue](https://huggingface.co/datasets/lex_glue) - Case law and legal queries
- **Programming:** [code_search_net](https://huggingface.co/datasets/code_search_net) - Code documentation queries
- **Psychology:** [FaithDial](https://github.com/McGill-NLP/FaithDial) - Mental health conversations

## Important: Dataset Preparation

### Avoid Topic Clustering

**Problem:** Ordered datasets can have topic clustering that produces narrow, repetitive training data.

**Example:** The `lex_glue` law dataset had:
- First 5,000 queries: Terms of Service (subscriptions, refunds, cancellations)
- Next 50,000 queries: Diverse case law

Testing with the first 50 queries produced poor quality negatives due to ToS contamination.

**Solution:**
1. ✅ Inspect first and last 100 samples for topic clustering
2. ✅ Use random sampling for testing if dataset has ordering bias
3. ✅ Skip contaminated sections: `tail -n +5210 dataset.jsonl > clean.jsonl`
4. ✅ Test with 20-50 queries first to verify quality before full generation

This saves hours of wasted GPU time.

## Production Features

### Streaming Writes
- Writes samples immediately to disk (no memory accumulation)
- Line-buffered I/O for real-time progress monitoring
- Handles millions of samples without OOM

### Checkpoint/Resume
```bash
# Resume from interruption
python3 generate_training_data.py ... --resume
```

Checkpoint tracks: queries processed, samples written, timestamp.

### Multi-GPU Scaling

```bash
--tensor-parallel 4  # Split model across 4 GPUs
```

Performance: 1 GPU ~6-8 hours, 4 GPUs ~1.5-2 hours for 44K queries.

### Error Handling
- Graceful LLM error recovery
- SIGINT/SIGTERM handling (saves checkpoint on Ctrl+C)
- Progress tracking with tqdm
- Detailed logging

## Technical Details

### MNR Loss with In-Batch Negatives

```python
# Compute similarity matrix [batch_size, batch_size]
similarity = cosine(anchors, positives.T) / temperature

# Labels: diagonal elements are correct positives
labels = [0, 1, 2, ..., batch_size-1]

# All other samples in batch serve as additional negatives
loss = CrossEntropy(similarity, labels)
```

This approach:
- Scales quadratically with batch size (more negatives = harder task)
- Is computationally efficient (single forward pass)
- Benefits from LLM-generated hard negatives being present in batches

### Pipeline Architecture

```
Unlabeled Queries (domain-specific questions)
    ↓
LLM Augmentation (Qwen 2.5 1.5B via vLLM)
  • 3 paraphrases per query (positives)
  • 2 hard negatives per query
  • Create triplets: anchor + positive + negative
    ↓
Training Triplets (~3x query count)
    ↓
LoRA Fine-Tuning (1 epoch, MNR loss, temp=0.05)
  • Base: all-MiniLM-L12-v2 (33.5M params)
  • LoRA: 147K trainable params (0.44%)
    ↓
Domain-Specific Cache Model (582KB adapter)
```

## References

- **Paper:** [Advancing Semantic Caching for LLMs with Domain-Specific Embeddings and Synthetic Data (arXiv:2504.02268v1)](https://arxiv.org/pdf/2504.02268v1)
- **Base Model:** [all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
- **LoRA:** [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **MNR Loss:** [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)

## License

- Code: Follow semantic-router repository license
- MedQuAD dataset: CC BY 4.0
