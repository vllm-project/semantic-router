---
name: signal-end-to-end
category: primary
description: Implements end-to-end signal changes spanning router config, signal extraction, CLI schema, optional bindings, dashboard surfaces, and E2E test coverage. Use when adding a new signal type, changing signal configuration or extraction logic, updating CLI schema for signal parameters, or modifying how signals are displayed in the dashboard.
---

# Signal End to End

## Trigger

- Add a new signal type or signal rule shape
- Change how an existing signal is configured, extracted, emitted, or displayed
- Touch router signal config plus Python CLI schema in the same feature

## Workflow

1. Read change surfaces, module boundaries, and playbooks for signal context
2. Modify signal config, extraction, CLI schema, and dashboard surfaces as needed
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to identify all impacted surfaces
4. Run `make agent-ci-gate CHANGED_FILES="..."` to validate signal contract alignment
5. Verify router config, signal extraction, and CLI schema stay aligned with relevant E2E coverage

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
