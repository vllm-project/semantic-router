---
name: routing-policy-runtime
category: fragment
description: Maintains matched-decision predicates, looper policy, and candidate-model selection behavior after signal extraction. Use when the task changes decision trees, thresholds, downstream model ranking, or other routing policy that runs after signals are already produced.
---

# Routing Policy Runtime

## Trigger

- The primary skill touches matched-decision predicates, thresholds, looper policy, or candidate-model selection after signal extraction

## Workflow

1. Read the Go router playbook and module boundaries for the post-signal policy seam
2. Modify matched-decision predicates, looper policy, or candidate-model selection logic
3. Run `make test-semantic-router` to validate routing-policy behavior
4. Verify matched-decision behavior, selection metadata, and tests describe the same routing outcome

## Must Read

- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)
- [docs/agent/playbooks/go-router.md](../../../../docs/agent/playbooks/go-router.md)

## Standard Commands

- `make test-semantic-router`

## Acceptance

- Matched-decision predicates, candidate-model selection, and targeted tests agree on the same routing-policy behavior
