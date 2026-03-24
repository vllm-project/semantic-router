---
name: plugin-runtime
category: fragment
description: Modifies post-decision plugin hooks that transform requests or responses in the routing pipeline. Use when adding, changing, or debugging plugin-controlled request filtering, response transformation, or middleware processing in the Go router.
---

# Plugin Runtime

## Trigger

- The primary skill touches plugin-controlled request or response processing

## Workflow

1. Read module boundaries and Go router playbook to understand plugin hook points
2. Modify plugin runtime code (request/response handlers, config hooks)
3. Run `make test-semantic-router` to verify behavior
4. Confirm plugin runtime behavior, config hooks, and tests stay aligned

## Must Read

- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)
- [docs/agent/playbooks/go-router.md](../../../../docs/agent/playbooks/go-router.md)

## Standard Commands

- `make test-semantic-router`

## Acceptance

- Plugin runtime behavior, config hooks, and tests stay aligned
