---
name: algorithm-selection-change
category: primary
description: Use when changing per-decision candidate-model selection after a decision matches.
---

# Algorithm Selection Change

## Trigger

- Change model-selection logic that runs after a decision matches
- Change per-decision candidate ranking, cost-aware routing, or latency-aware model choice

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
