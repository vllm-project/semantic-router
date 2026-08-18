# Hallucination-Span Classifier Training

This standalone pipeline trains a token classifier over a context-and-answer
pair. Each answer token is labelled as supported or hallucinated.

## Install

```bash
pip install -r requirements.txt
```

## Prepare Data

Provide RAGTruth and, optionally, DART or E2E augmentation files:

```bash
python prepare_data.py \
  --ragtruth-path /path/to/ragtruth_data.json \
  --dart-path /path/to/dart_spans.json \
  --e2e-path /path/to/e2e_spans.json \
  --output-dir data
```

`--download-augmentation` can download supported DART/E2E inputs instead of
using local paths. Review dataset licenses and the generated train/dev/test
counts before training.

## Train

```bash
python finetune.py \
  --train-path data/train.json \
  --dev-path data/dev.json \
  --test-path data/test.json \
  --output-dir output/haldetect-32k \
  --batch-size 8 \
  --learning-rate 1e-5 \
  --epochs 6
```

`run_training.sh` wraps preparation and training in the environment expected by
the script. Read it before use so its paths and downloads match your machine.

## Evaluate the Right Boundary

Token metrics alone do not establish answer-level reliability. Report
token- and span-level precision/recall, performance by answer length and domain,
the context truncation policy, and examples of unsupported spans the model
misses. A trained checkpoint is not automatically loaded by semantic-router;
runtime integration and configuration must be validated separately.
