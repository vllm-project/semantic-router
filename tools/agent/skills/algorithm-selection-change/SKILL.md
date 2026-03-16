---
name: algorithm-selection-change
category: primary
description: Modifies candidate-model selection logic that runs after a routing decision matches, including ranking, cost-aware routing, and latency-aware model choice. Use when changing how the router selects which model serves a matched decision, updating candidate ranking algorithms, or adjusting model cost/latency trade-offs.
---

# Algorithm Selection Change

## Trigger

- Change model-selection logic that runs after a decision matches
- Change per-decision candidate ranking, cost-aware routing, or latency-aware model choice

## Workflow

1. Read change surfaces, module boundaries, and Go router playbook for context
2. Modify algorithm selection logic (ranking, cost, latency routing)
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to identify impacted surfaces
4. Run `make agent-ci-gate CHANGED_FILES="..."` to validate all constraints pass
5. Verify selection behavior is covered by targeted tests and affected E2E profiles

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)
- [docs/agent/playbooks/go-router.md](../../../../docs/agent/playbooks/go-router.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Candidate-model selection behavior is covered by targeted tests and affected E2E profiles
- Algorithm logic stays downstream of the matched decision instead of leaking into signal or plugin code
