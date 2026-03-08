---
name: header-contract-change
category: primary
description: Use when adding or changing x-vsr request/response headers and the downstream UI/display contract.
---

# Header Contract Change

## Trigger

- Add a new `x-vsr-*` header
- Rename, remove, or change the meaning of an existing router header
- Change how dashboard/playground reveals routing metadata

## Required Surfaces

- `response_headers`

## Conditional Surfaces

- `dashboard_config_ui`
- `playground_reveal`
- `topology_visualization`
- `local_e2e`
- `ci_e2e`

## Stop Conditions

- The header semantics are not stable enough to expose to users

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/feature-complete-checklist.md](../../../../docs/agent/feature-complete-checklist.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`

## Acceptance

- Header constants, emission, and UI allowlists remain aligned
- Relevant tests cover the new or changed header contract
