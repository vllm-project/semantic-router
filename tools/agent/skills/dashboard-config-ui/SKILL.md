---
name: dashboard-config-ui
category: fragment
description: Dashboard config editor, config display, and mutation-flow details.
---

# Dashboard Config UI

## Trigger

- The primary skill touches dashboard config forms or router-backed config display

## Required Surfaces

- `dashboard_config_ui`

## Conditional Surfaces

- `topology_visualization`
- `playground_reveal`

## Stop Conditions

- Dashboard serialization conflicts with router or CLI contract

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)

## Standard Commands

- `make dashboard-check`
- `make agent-ci-gate CHANGED_FILES="..."`

## Acceptance

- Dashboard config editing remains aligned with router and CLI schema
