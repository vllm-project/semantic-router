---
name: signal-end-to-end
category: primary
description: Use when adding or changing a signal that spans router config, signal extraction, CLI schema, optional bindings, dashboard surfaces, and E2E.
---

# Signal End to End

## Trigger

- Add a new signal type or signal rule shape
- Change how an existing signal is configured, extracted, emitted, or displayed
- Touch router signal config plus Python CLI schema in the same feature

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)
- [docs/agent/playbooks/go-router.md](../../../../docs/agent/playbooks/go-router.md)
- [docs/agent/playbooks/vllm-sr-cli-docker.md](../../../../docs/agent/playbooks/vllm-sr-cli-docker.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Router config, signal extraction, and Python CLI schema stay aligned for the signal contract
- Any user-visible signal metadata updates the relevant header or dashboard surfaces
- Relevant E2E coverage is added or updated when behavior changes
