---
name: training-stack-change
category: primary
description: Modifies training-stack workflows, selector or embedding pipelines, evaluation artifacts, or runtime-facing outputs under src/training. Use when changing model selection training, embedding pipelines, evaluation scripts, experiments, or other training outputs that feed runtime behavior.
---

# Training Stack Change

## Trigger

- Change training workflows, scripts, docs, or runtime-facing outputs under `src/training/**`
- Change selector, embedding, evaluation, or experiment artifacts that feed runtime behavior

## Workflow

1. Read change surfaces and the repo map for the current training stack context
2. Modify the training workflow, artifact expectation, or runtime-facing output
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to verify surface alignment
4. Run `make agent-ci-gate CHANGED_FILES="..."` to validate all constraints
5. Record any runtime-facing artifact contract mismatches as indexed debt entries

## Gotchas

- `src/training/**` is no longer classifier-only; do not force selector, embedding, eval, or experiment work through classifier language that hides the real contract.
- Training-only edits still change runtime behavior when exported artifacts, defaults, or evaluation gates shift.

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/repo-map.md](../../../../docs/agent/repo-map.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`

## Acceptance

- Training workflow changes keep scripts, docs, and artifact expectations aligned
- Runtime-facing artifact contract changes are either updated in code or recorded as tracked debt through the matching indexed debt entry
