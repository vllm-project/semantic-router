---
name: training-model-classifier
category: fragment
description: Manages model-classifier fine-tuning workflows, training scripts, and runtime-facing artifact expectations. Use when modifying training configurations, updating training data pipelines, changing model hyperparameters, or adjusting how training artifacts are consumed by the router runtime.
---

# Training Model Classifier

## Trigger

- The primary skill touches model-classifier training scripts, docs, or artifact expectations

## Workflow

1. Read the model classifier README and tech debt docs for current training setup
2. Modify training scripts, configurations, or artifact expectations
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to verify alignment
4. Confirm training scripts, docs, and artifact expectations stay aligned

## Must Read

- [../../../../src/training/model_classifier/README.md](../../../../src/training/model_classifier/README.md)
- [docs/agent/tech-debt-register.md](../../../../docs/agent/tech-debt-register.md)
- [docs/agent/tech-debt/README.md](../../../../docs/agent/tech-debt/README.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Training scripts, docs, and artifact expectations stay aligned
