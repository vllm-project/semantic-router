---
title: Train the mmBERT-32K Safety Classifiers
sidebar_label: Safety Classifiers
---

# Train the mmBERT-32K safety classifiers

The safety workflow uses two classifiers in sequence:

```text
prompt -> Level 1: safe / unsafe
                    |
                    +-- safe   -> continue normal routing
                    +-- unsafe -> Level 2: one of nine hazard outputs
```

Use Level 1 when a binary policy decision is sufficient. Add Level 2 when an
unsafe request must be routed, logged, or handled differently by hazard type.
The second model is not intended to run on requests that Level 1 accepts as
safe.

## Published artifact architecture

Both artifacts in the current collection are PEFT LoRA adapters for
`ModernBertForSequenceClassification` on
[`jhu-clsp/mmBERT-base`](https://huggingface.co/jhu-clsp/mmBERT-base). They
truncate inputs to 512 tokens and adapt four attention/MLP projection groups:
`attn.Wqkv`, `attn.Wo`, `mlp.Wi`, and `mlp.Wo`.

| Task | Head | Published artifact shape |
| --- | --- | --- |
| Level 1 | Two-class sequence head | [`mmbert-safety-binary-merged`](https://huggingface.co/llm-semantic-router/mmbert-safety-binary-merged), PEFT adapter |
| Level 2 | Nine-class sequence head | [`mmbert-safety-binary-hazard`](https://huggingface.co/llm-semantic-router/mmbert-safety-binary-hazard), PEFT adapter |

The Level 1 name ends in `-merged`, but its published files contain
`adapter_model.safetensors` and `adapter_config.json`, not standalone base
weights. Load it with the base model declared by its adapter configuration.

## Current 32K training architecture

The checked workflow trains successor artifacts on
[`mmbert-32k-yarn`](https://huggingface.co/llm-semantic-router/mmbert-32k-yarn)
while preserving the same two heads, labels, data policy, LoRA targets, and
512-token safety input limit. It can export both adapter and full merged shapes
for either level and verifies their logits before release.

Do not attach an existing `mmBERT-base` adapter to the 32K base. Use the base
declared by the artifact for existing checkpoints; use the 32K base only for a
new run produced by the current training contract.

## Labels

Level 1 uses `safe` and `unsafe`. Level 2 preserves the following nine-output
compatibility contract:

| ID | Meaning |
| --- | --- |
| `S1_violent_crimes` | Violent crimes |
| `S2_nonviolent_crimes` | Non-violent crimes |
| `S3_sex_crimes` | Sex-related crimes |
| `S5_weapons_cbrne` | Weapons and CBRNE |
| `S6_self_harm` | Self-harm |
| `S7_hate` | Hate |
| `S8_specialized_advice` | Specialized advice |
| `S9_privacy` | Privacy |
| `S13_misinformation` | Misinformation |

This order is versioned as `legacy-9-v1`. Treat the strings and numeric order
as an API: changing either requires a migration of router policy and stored
evaluation data.

## Data preparation

The workflow uses prompt labels from AEGIS 2.0 plus a synthetic safety dataset.
Response and refusal variants are excluded. Preparation normalizes text for
deduplication, removes empty or redacted records, gives held-out splits
precedence over training data, and drops duplicate groups with conflicting
labels.

The checked data contract creates:

- Level 1: 10,000 training prompts per binary label;
- Level 2: 2,000 training prompts per hazard label, with deterministic
  oversampling only where a class is short.

Validation and test splits keep their natural AEGIS distribution. For prompts
with multiple mapped hazards, the first mapped source category supplies the
single training label while all mapped hazards remain available for stricter
evaluation.

Prepare the data once before distributed training:

```bash
python -m src.training.model_classifier.safety_classifier.data prepare \
  --contract src/training/model_classifier/safety_classifier/configs/training-v1.json \
  --output-dir /artifacts/data
```

The command verifies pinned input revisions and file checksums and writes
materialized splits under `/artifacts/data/level1` and
`/artifacts/data/level2`.

## Training method

Both tasks use LoRA rank 32, alpha 64, dropout 0.1, AdamW, a linear schedule,
10% warmup, weight decay `0.01`, BF16, seed 42, and early stopping with patience
3. The checked eight-process topology uses per-device batch 8 for global batch
64 and trains for at most 10 epochs.

```bash
torchrun --standalone --nproc_per_node=8 \
  -m src.training.model_classifier.safety_classifier.train \
  --task level1 \
  --expected-world-size 8 \
  --data-dir /artifacts/data \
  --output-dir /artifacts/runs/level1

torchrun --standalone --nproc_per_node=8 \
  -m src.training.model_classifier.safety_classifier.train \
  --task level2 \
  --expected-world-size 8 \
  --data-dir /artifacts/data \
  --output-dir /artifacts/runs/level2
```

Use `--max-steps 2` for a short accelerator smoke. A run that overrides the
checked contract is still useful for experimentation, but record the resolved
configuration with its metrics rather than treating it as the standard
release recipe.

## Evaluate and export

Choose checkpoints by macro F1 and inspect per-class precision and recall. For
Level 1, false negatives and false positives should be reported separately.
For Level 2, include the confusion matrix and strict multi-hazard recall so a
high-frequency class cannot hide a weak hazard boundary.

```bash
python -m src.training.model_classifier.safety_classifier.evaluate \
  --task level1 \
  --model /artifacts/runs/level1/adapter \
  --artifact-type adapter \
  --data /artifacts/data/level1/test.jsonl \
  --output-dir /artifacts/runs/level1/evaluation

python -m src.training.model_classifier.safety_classifier.export \
  --task level1 \
  --run-root /artifacts/runs/level1 \
  --merged-dir /artifacts/runs/level1-merged
```

Repeat with `--task level2` for the hazard model. Export compares adapter and
merged logits on fixed examples, checks prediction identity within the
configured tolerance, and writes checksums and label metadata.

The complete CLI, environment, and release commands are in the
[workflow README](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_classifier/safety_classifier).
