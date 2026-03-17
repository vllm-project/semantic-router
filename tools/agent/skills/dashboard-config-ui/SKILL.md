---
name: dashboard-config-ui
category: fragment
description: Modifies dashboard configuration editor forms, config display components, and mutation-flow logic that connects the UI to the router config contract. Use when editing dashboard config forms, updating how config values are displayed, or changing the mutation flow between the dashboard UI and the router/CLI schema.
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

## Workflow

1. Read change surfaces doc to understand config UI dependencies
2. Modify dashboard config forms, display components, or mutation-flow logic
3. Run `make dashboard-check` to validate UI consistency
4. Run `make agent-ci-gate CHANGED_FILES="..."` to verify alignment with router and CLI schema

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)

## Standard Commands

- `make dashboard-check`
- `make agent-ci-gate CHANGED_FILES="..."`

## Acceptance

- Dashboard config editing remains aligned with router and CLI schema
