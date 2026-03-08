---
name: cross-stack-bugfix
category: primary
description: Use when fixing a multi-surface issue that does not map cleanly to a narrower project-level skill.
---

# Cross Stack Bugfix

## Trigger

- A bug spans multiple layers and no narrower primary skill clearly applies
- The fix needs coordinated changes across runtime, CLI, UI, or test surfaces

## Required Surfaces

- `local_e2e`
- `ci_e2e`

## Conditional Surfaces

- `router_config_contract`
- `router_runtime`
- `native_binding`
- `response_headers`
- `python_cli_schema`
- `dashboard_config_ui`
- `topology_visualization`
- `playground_reveal`
- `dsl_crd`
- `docs_examples`
- `local_smoke`

## Stop Conditions

- The bug spans multiple contracts but the owning behavior is still ambiguous

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/feature-complete-checklist.md](../../../../docs/agent/feature-complete-checklist.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- The final report explicitly names impacted surfaces and intentionally skipped conditional surfaces
