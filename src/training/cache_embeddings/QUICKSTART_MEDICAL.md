# Quick Start: Medical Domain Cache Embeddings

Train a LoRA adapter for medical query caching in 3 simple steps.

## Prerequisites

```bash
pip install -r requirements.txt
```

## Step 1: Prepare Medical Data

Download and prepare the MedQuAD dataset (47K medical questions from NIH, CDC, FDA):

```bash
python3 src/training/cache_embeddings/prepare_medical_data.py \
  --output data/cache_embeddings/medical/unlabeled_queries.jsonl
```

**Output:** `~47,000 medical questions`

## Step 2: Generate Training Triplets

Use vLLM to generate paraphrases and hard negatives:

```bash
python3 src/training/cache_embeddings/generate_training_data.py \
  --input data/cache_embeddings/medical/unlabeled_queries.jsonl \
  --output data/cache_embeddings/medical/triplets.jsonl \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --domain medical \
  --paraphrases 3 \
  --negatives 2 \
  --batch-size 32 \
  --tensor-parallel 4
```

**Requirements:** 4x GPU (A10G or better)
**Time:** ~2-3 hours
**Output:** `~230,000 training triplets`

**Alternative (CPU-only, small test):**
```bash
# Test with 100 queries first
python3 src/training/cache_embeddings/generate_training_data.py \
  --input data/cache_embeddings/medical/unlabeled_queries.jsonl \
  --output data/cache_embeddings/medical/triplets_test.jsonl \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --domain medical \
  --paraphrases 3 \
  --negatives 2 \
  --max-queries 100
```

## Step 3: Train LoRA Adapter

Train the LoRA adapter using Multiple Negatives Ranking loss:

```bash
python3 src/training/cache_embeddings/lora_trainer.py \
  --train-data data/cache_embeddings/medical/triplets.jsonl \
  --base-model sentence-transformers/all-MiniLM-L12-v2 \
  --output models/medical-cache-lora \
  --epochs 1 \
  --batch-size 32 \
  --lr 2e-5 \
  --temperature 0.05
```

**Time:** ~30 minutes (GPU) or ~2 hours (CPU)
**Output:** LoRA adapter (~582 KB)

## Step 4: Test the Model

```bash
python3 src/training/cache_embeddings/test_medical_model.py
```

**Expected Results:**
- Baseline margin: ~0.17
- LoRA margin: ~0.21
- **Improvement: +21.4%**

## Use in Production

```python
from sentence_transformers import SentenceTransformer
from peft import PeftModel

# Load base model
base_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

# Apply LoRA adapter
base_model[0].auto_model = PeftModel.from_pretrained(
    base_model[0].auto_model,
    "models/medical-cache-lora"
)

# Use for caching
query = "What are the symptoms of diabetes?"
embedding = base_model.encode(query)
```

## One-Command Training (AWS)

Use the automated training script:

```bash
./src/training/cache_embeddings/train-domain.sh medical
```

This will:
1. Launch AWS instance (g5.12xlarge with 4x A10G GPUs)
2. Download MedQuAD dataset
3. Generate triplets with vLLM
4. Train LoRA adapter
5. Download model
6. Cleanup AWS resources

**Cost:** ~$10-15 total

## Pre-trained Model

Download the pre-trained medical LoRA adapter:

```bash
huggingface-cli download vllm-project/medical-cache-lora
```

Or use directly in code:

```python
from sentence_transformers import SentenceTransformer
from peft import PeftModel

base_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
base_model[0].auto_model = PeftModel.from_pretrained(
    base_model[0].auto_model,
    "vllm-project/medical-cache-lora"
)
```

## Dataset Details

**MedQuAD (Medical Question Answering Dataset)**
- Source: https://github.com/abachaa/MedQuAD
- Size: 47,457 medical Q&A pairs
- License: Public domain (NIH/NLM)
- Quality: HIGH - Curated from NIH, CDC, FDA, WHO

**Sources:**
- NIH Senior Health
- GARD (Genetic and Rare Diseases Information Center)
- CDC (Centers for Disease Control and Prevention)
- FDA (Food and Drug Administration)
- NIDDK (National Institute of Diabetes and Digestive and Kidney Diseases)
- NINDS (National Institute of Neurological Disorders and Stroke)
- Plus 8 more trusted medical institutions

## Performance

| Metric | Baseline | LoRA | Improvement |
|--------|----------|------|-------------|
| Avg Positive Similarity | 0.835 | 0.853 | +2.2% |
| Avg Negative Similarity | 0.663 | 0.642 | -3.2% |
| Margin | 0.172 | 0.211 | **+21.4%** |

Better margin = better cache hit/miss discrimination!

## Next Steps

- Try other domains: [programming](domains/programming.yaml), [legal](domains/legal.yaml), [financial](domains/financial.yaml)
- See [full documentation](docs/README.md)
- Read the [research blog post](docs/blog.md)
