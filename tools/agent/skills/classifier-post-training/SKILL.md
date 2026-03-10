---
name: classifier-post-training
category: primary
description: Use when changing model-classifier fine-tuning, post-training data flows, or training artifacts that feed runtime behavior.
---

# Classifier Post Training

## Trigger

- Change model-classifier fine-tuning scripts or training docs
- Change runtime-facing classifier artifacts or post-training workflow expectations

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/tech-debt-register.md](../../../../docs/agent/tech-debt-register.md)
- [docs/agent/tech-debt/README.md](../../../../docs/agent/tech-debt/README.md)
- [src/training/model_classifier/README.md](../../../../src/training/model_classifier/README.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`

## Acceptance

- Training workflow changes keep scripts, docs, and artifact expectations aligned
- Runtime-facing artifact contract changes are either updated in code or recorded as tracked debt through the matching indexed debt entry
