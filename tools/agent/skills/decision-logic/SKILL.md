---
name: decision-logic
category: fragment
description: Implements boolean decision predicates, thresholds, and control-logic that combine signals and route conditions into routing decisions. Use when modifying conditional routing rules, adding decision predicates, or changing threshold logic in the Go router.
---

# Decision Logic

## Trigger

- The primary skill touches decision predicates, thresholds, or control-logic code

## Workflow

1. Read module boundaries and Go router playbook to understand predicate structure
2. Modify decision predicates, thresholds, or control-logic code
3. Run `make test-semantic-router` to validate decision behavior
4. Verify decision predicates, thresholds, and tests describe the same behavior

## Must Read

- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)
- [docs/agent/playbooks/go-router.md](../../../../docs/agent/playbooks/go-router.md)

## Standard Commands

- `make test-semantic-router`

## Acceptance

- Decision predicates, thresholds, and tests describe the same behavior
