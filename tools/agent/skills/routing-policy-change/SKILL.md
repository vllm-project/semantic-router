---
name: routing-policy-change
category: primary
description: Modifies routing policy after signal extraction, including matched-decision logic, candidate-model selection, and downstream looper behavior. Use when changing decision predicates, thresholds, priorities, model ranking, cost or latency routing, or other post-signal routing policy.
---

# Routing Policy Change

## Trigger

- Change boolean decision logic after signal extraction
- Change candidate-model selection, ranking, cost routing, or latency routing
- Change looper behavior that belongs to matched-decision policy instead of signal extraction or plugin hooks

## Workflow

1. Read change surfaces, module boundaries, and the Go router playbook for routing-policy context
2. Modify decision predicates, downstream model selection, or matched-decision looper behavior
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to identify impacted surfaces
4. Run `make agent-ci-gate CHANGED_FILES="..."` to validate all affected constraints
5. Verify routing-policy behavior is covered by targeted tests and affected E2E expectations

## Gotchas

- Keep signal extraction upstream; do not smuggle classifier changes into this primary unless the signal contract itself is changing.
- Keep plugin hooks separate; looper changes that only support matched-decision policy belong here, but request or response body plugins still belong to the plugin flow.
- Decision predicates and candidate-model ranking can change user-visible routing even when the patch looks small, so they are not refactors by default.

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)
- [docs/agent/playbooks/go-router.md](../../../../docs/agent/playbooks/go-router.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Decision predicates and downstream candidate-model selection stay aligned with the same routing-policy intent
- Targeted tests and affected E2E coverage are updated when routing-policy behavior changes
- Routing-policy logic stays downstream of signal extraction and separate from plugin-only processing
