---
name: decision-logic-change
category: primary
description: Modifies boolean decision predicates, thresholds, gates, and priority-driven routing branches that combine signals into routing decisions. Use when changing how signals are evaluated into boolean logic, adding or removing decision gates, or adjusting threshold-based routing behavior.
---

# Decision Logic Change

## Trigger

- Change decision predicates, thresholds, gates, or priority-driven routing branches
- Change how signals are combined into boolean control logic

## Workflow

1. Read change surfaces, module boundaries, and Go router playbook for decision logic context
2. Modify decision predicates, thresholds, gates, or priority-driven routing branches
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to identify impacted surfaces
4. Run `make agent-ci-gate CHANGED_FILES="..."` to validate all constraints
5. Verify boolean decision behavior is covered by targeted tests and affected E2E profiles

## Gotchas

- Do not smuggle signal-extraction changes into the decision layer unless the routing contract itself is changing.
- Threshold and priority tweaks are behavior changes, not refactors; they still need explicit tests and affected E2E coverage.

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)
- [docs/agent/playbooks/go-router.md](../../../../docs/agent/playbooks/go-router.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Boolean decision behavior is covered by targeted tests and affected E2E profiles
- Signal extraction and decision ownership stay separated by layer
