---
name: training-stack-runtime
category: fragment
description: Modifies training-stack workflows, selector or embedding pipelines, evaluation scripts, and runtime-facing artifact expectations. Use when a primary skill touches src/training, tools/make/models.mk, or training docs for runtime-fed artifacts.
---

# Training Stack Runtime

## Trigger

- The primary skill touches training workflows, scripts, docs, or runtime-facing artifact expectations under `src/training/**`

## Workflow

1. Read the repo map for the affected training area
2. Modify training scripts, configurations, or artifact expectations
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to verify alignment
4. Confirm training scripts, docs, and artifact expectations stay aligned

## Must Read

- [docs/agent/repo-map.md](../../../../docs/agent/repo-map.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Training scripts, docs, and artifact expectations stay aligned
