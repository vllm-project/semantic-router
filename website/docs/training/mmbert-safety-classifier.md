---
title: Rebuild the mmBERT-32K safety classifiers
sidebar_position: 4
---

# Rebuild the mmBERT-32K safety classifiers

Semantic Router provides a reproducible source workflow for its hierarchical
mmBERT-32K prompt-safety classifiers:

1. Level 1 predicts `safe` or `unsafe`.
2. Level 2 assigns unsafe prompts to one of nine historical hazard outputs.

The workflow lives in
[`src/training/model_classifier/safety_classifier`](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_classifier/safety_classifier).

## Why these are new checkpoints

The historical adapters now named `mmbert-safety-binary-merged` and
`mmbert-safety-binary-hazard` were originally published as
`mlcommons-safety-classifier-level1-binary` and
`mlcommons-safety-classifier-level2-hazard`. Their cards describe an older
`jhu-clsp/mmBERT-base` LoRA run, but no producing trainer or exact split was
published.

The in-tree workflow reconstructs those tasks with the same LoRA rank, alpha,
dropout, target modules, 512-token training context, and reported high-level
hyperparameters. It changes the base to the pinned
`llm-semantic-router/mmbert-32k-yarn` checkpoint used by the current classifier
family. The resulting models are new experiments, not renamed or rebased old
adapters.

## Reproducibility boundary

The versioned contract pins the base and datasets by immutable revision and
pins each raw input file by SHA-256. Dataset preparation is deterministic and
records exclusions, conflicts, split de-duplication, sampling, per-class
counts, and final split hashes.

Historical facts and reconstruction choices are intentionally distinct:

| Contract element | Status |
| --- | --- |
| LoRA rank 32, alpha 64, dropout 0.1 | Recovered from adapter metadata |
| Four ModernBERT attention/MLP targets | Recovered from adapter metadata |
| 10 epochs, global batch 64, learning rate 3e-4 | Recovered from model cards |
| Prompt-only training and max length 512 | Controlled reconstruction choice |
| Seed 42, warmup ratio 0.1, weight decay 0.01 | Controlled reconstruction choice |
| Exact split, de-duplication, and multi-label collapse | New deterministic contract |

## Legacy nine-class taxonomy

The Level 2 output order is retained for compatibility, but it is named
`legacy-9-v1`. Some historical `S` identifiers do not match the canonical
13-category numbering used by newer MLCommons-derived taxonomies. In
particular, the synthetic dataset's raw S6/S7/S9/S11 IDs mean specialized
advice, privacy, indiscriminate weapons, and self-harm; the legacy model mapped
those semantics to different output IDs.

The source taxonomy and model-output taxonomy are separate mappings in code.
Unknown source categories fail preparation rather than being silently dropped.
This preserves the current output ABI while making a future canonical-taxonomy
migration possible.

## Artifact contract

Each task publishes two unambiguous artifact shapes:

- `-lora`: PEFT adapter plus the sequence-classification head;
- `-merged`: full ModernBERT sequence-classification weights.

Export compares adapter and merged logits on fixed fixtures, checks prediction
identity, and writes file checksums. The release command downloads each remote
revision again, verifies those checksums, reloads the model, and runs one
inference before recording the publication receipt.

For commands, environment pins, data rules, and evaluation metrics, see the
[workflow README](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_classifier/safety_classifier).
