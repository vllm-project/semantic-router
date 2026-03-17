---
name: classifier-post-training
category: primary
description: Modifies model-classifier fine-tuning scripts, post-training data flows, and training artifacts that feed the router runtime. Use when changing training configurations, updating post-training data pipelines, modifying classifier artifact formats, or adjusting how training outputs are consumed by runtime components.
---

# Classifier Post Training

## Trigger

- Change model-classifier fine-tuning scripts or training docs
- Change runtime-facing classifier artifacts or post-training workflow expectations

## Workflow

1. Read change surfaces, model classifier README, and tech debt docs for context
2. Modify fine-tuning scripts, post-training workflows, or artifact expectations
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to verify surface alignment
4. Run `make agent-ci-gate CHANGED_FILES="..."` to validate all constraints
5. Record any runtime-facing artifact contract mismatches as indexed debt entries

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
