# Semantic-Cache Embedding Training

These scripts train LoRA adapters that make an embedding model separate
paraphrases from related-but-different queries. That distinction is useful for
semantic caches, where an overly broad match can return the wrong response.

This directory produces training and evaluation artifacts. Loading a LoRA
adapter in the router is a separate runtime capability; do not assume a trained
adapter is active because it appears in a config file.

## Data Contract

Training data is JSONL with one triplet per line:

```json
{"anchor":"How is diabetes diagnosed?","positive":"Which tests diagnose diabetes?","negative":"How is hypertension treated?"}
```

The positive should be safe to reuse as the same cache intent. The negative
should be topically close but require a different answer. Review generated
triplets before training; fluent synthetic text can still contain label errors.

## Generate Triplets

`generate_training_data.py` uses a local vLLM-supported generation model and
the domain prompts in `domains/prompts.yaml`.

```bash
python src/training/model_embeddings/cache_embeddings/generate_training_data.py \
  --input data/medical_queries.jsonl \
  --domain medical \
  --output data/medical_triplets.jsonl \
  --max-queries 1000
```

Input lines must contain `{"query":"..."}`. Use `--resume` to continue from the
script's checkpoint after an interrupted generation job.

## Train an Adapter

```bash
python src/training/model_embeddings/cache_embeddings/lora_trainer.py \
  --train-data data/medical_triplets.jsonl \
  --base-model sentence-transformers/all-MiniLM-L12-v2 \
  --output models/medical-cache-lora \
  --epochs 1
```

Use a separate validation file with `--val-data` when tuning hyperparameters.
Record the base model with the adapter; a LoRA checkpoint cannot be interpreted
without it.

To combine domains, concatenate reviewed triplet files and train one adapter.
Compare that adapter with domain-specific alternatives on the same held-out
sets before choosing a deployment strategy.

## Evaluate

`evaluate_multi_domain.py` compares a LoRA adapter with the MiniLM-L12 baseline
on medical, law, and programming triplets:

```bash
python src/training/model_embeddings/cache_embeddings/evaluate_multi_domain.py \
  --lora-path models/medical-cache-lora \
  --sample-size 2000
```

The main metric is the mean similarity margin:

```text
mean(anchor, positive) - mean(anchor, negative)
```

A larger margin is useful, but it does not select a safe cache threshold on its
own. Also measure false reuse and missed reuse at the threshold used by the
router, with a held-out and representative workload.

## Files

| File | Purpose |
|---|---|
| `domains/prompts.yaml` | generation prompts and supported domain aliases |
| `generate_training_data.py` | create paraphrase and hard-negative triplets |
| `lora_trainer.py` | train an adapter with multiple-negatives ranking loss |
| `evaluate_multi_domain.py` | compare baseline and adapter margins |
| `test_lora_model.py` | inspect one trained adapter interactively |

Published model links and measured results belong in the model card for the
specific artifact, including its dataset, split, threshold analysis, and known
domains. They are intentionally not duplicated here.
