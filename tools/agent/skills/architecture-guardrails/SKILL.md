---
name: architecture-guardrails
category: fragment
description: Structural limits, dependency boundaries, interface placement, and preferred composition-oriented design patterns.
---

# Architecture Guardrails

## Trigger

- Load this fragment for any non-trivial code change

## Required Surfaces

- `router_runtime`
- `python_cli_schema`
- `dashboard_config_ui`

## Conditional Surfaces

- `native_binding`
- `topology_visualization`

## Stop Conditions

- The edit needs an exception to current structure or dependency rules

## Must Read

- [docs/agent/architecture-guardrails.md](../../../../docs/agent/architecture-guardrails.md)

- [tools/agent/structure-rules.yaml](../../structure-rules.yaml)
- [tools/agent/scripts/structure_check.py](../../scripts/structure_check.py)
- language-specific lint steps in the agent gate

## Standard Commands

- `make agent-lint CHANGED_FILES="..."`

## Acceptance

- Changed code passes structure rules and stays modular
