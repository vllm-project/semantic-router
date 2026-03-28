---
name: signal-runtime
category: fragment
description: Modifies signal extraction and text-understanding runtime code that produces facts consumed by decision logic. Use when adding new signals, changing classifier runtime behavior, or updating how the router extracts features from input text.
---

# Signal Runtime

## Trigger

- The primary skill touches signal extraction or classifier runtime code

## Workflow

1. Read module boundaries and Go router playbook to understand signal extraction paths
2. Modify signal extraction or classifier runtime code
3. Run `make test-semantic-router` to verify extracted facts match expectations
4. Confirm signal extraction code, emitted facts, and targeted tests stay aligned

## Must Read

- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)
- [docs/agent/playbooks/go-router.md](../../../../docs/agent/playbooks/go-router.md)

## Standard Commands

- `make test-semantic-router`

## Acceptance

- Signal extraction code, emitted facts, and targeted tests stay aligned
