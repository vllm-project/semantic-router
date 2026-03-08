---
name: router-runtime
category: fragment
description: Runtime implementation details for signal evaluation, decision logic, extproc behavior, and emitted routing metadata.
---

# Router Runtime

## Trigger

- The primary skill touches runtime evaluation or extproc behavior

## Required Surfaces

- `router_runtime`

## Conditional Surfaces

- `response_headers`
- `local_e2e`
- `ci_e2e`

## Stop Conditions

- Runtime behavior changes require a contract change outside the router that is not defined

## Must Read

- [docs/agent/playbooks/go-router.md](../../../../docs/agent/playbooks/go-router.md)

## Standard Commands

- `make agent-ci-gate CHANGED_FILES="..."`
- `make test-semantic-router`

## Acceptance

- Runtime behavior, tests, and emitted metadata stay aligned
