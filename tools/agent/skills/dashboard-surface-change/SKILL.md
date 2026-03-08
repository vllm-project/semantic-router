---
name: dashboard-surface-change
category: primary
description: Use when changing dashboard config, topology, or playground surfaces that reflect router behavior.
---

# Dashboard Surface Change

## Trigger

- Change config editing UI, topology visualization, or playground reveal/display
- Change frontend handling of routing metadata or router-backed config

## Required Surfaces

- `dashboard_config_ui`

## Conditional Surfaces

- `topology_visualization`
- `playground_reveal`
- `response_headers`
- `docs_examples`

## Stop Conditions

- The dashboard change depends on a backend/router contract that is not yet defined

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/feature-complete-checklist.md](../../../../docs/agent/feature-complete-checklist.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`

## Acceptance

- Dashboard surfaces remain aligned with router/CLI contracts
- User-visible routing metadata stays consistent across config, topology, and playground
