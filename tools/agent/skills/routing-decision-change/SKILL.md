---
name: routing-decision-change
category: primary
description: Use when changing decision evaluation, model selection, or routing policy behavior in the runtime path.
---

# Routing Decision Change

## Trigger

- Change decision priority, rule evaluation, or fallback logic
- Change model selection or routing strategy behavior
- Change runtime reasoning or selection metadata

## Required Surfaces

- `router_runtime`
- `local_e2e`
- `ci_e2e`

## Conditional Surfaces

- `response_headers`
- `docs_examples`

## Stop Conditions

- The intended routing policy is unclear or lacks testable expectations

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/playbooks/go-router.md](../../../../docs/agent/playbooks/go-router.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Runtime tests and affected E2E profiles reflect the routing change
- Any new routing metadata is exposed consistently
