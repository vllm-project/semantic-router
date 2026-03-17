---
name: algorithm-selection
category: fragment
description: Implements candidate-model selection logic that runs after a routing decision matches, including model ranking, cost-aware routing, and latency-aware model choice. Use when reading or modifying how the router picks which model serves a matched decision.
---

# Algorithm Selection

## Trigger

- The primary skill touches candidate-model selection after a decision matches

## Workflow

1. Read module boundaries and Go router playbook to understand selection entry points
2. Modify candidate-model selection logic (ranking, cost, latency)
3. Run `make test-semantic-router` to validate selection behavior
4. Verify algorithm code, selection metadata, and tests agree on selection behavior

## Must Read

- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)
- [docs/agent/playbooks/go-router.md](../../../../docs/agent/playbooks/go-router.md)

## Standard Commands

- `make test-semantic-router`

## Acceptance

- Algorithm code, selection metadata, and tests agree on selection behavior
