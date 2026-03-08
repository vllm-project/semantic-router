---
name: config-schema-evolution
category: primary
description: Use when changing a config contract that must stay aligned across router config, Python CLI schema, dashboard config UI, and docs/examples.
---

# Config Schema Evolution

## Trigger

- Add, rename, or change a config field
- Change config compatibility or normalization behavior
- Update config contract in router plus CLI/dashboard

## Required Surfaces

- `router_config_contract`
- `python_cli_schema`
- `dashboard_config_ui`
- `docs_examples`

## Conditional Surfaces

- `dsl_crd`
- `topology_visualization`
- `playground_reveal`

## Stop Conditions

- Compatibility policy between router, CLI, and dashboard is not decided

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/playbooks/go-router.md](../../../../docs/agent/playbooks/go-router.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Router, CLI, dashboard, and docs/examples describe the same schema
- DSL/CRD paths are updated when the schema is externally supported there
