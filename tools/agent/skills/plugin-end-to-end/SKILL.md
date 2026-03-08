---
name: plugin-end-to-end
category: primary
description: Use when changing plugin config, plugin runtime behavior, plugin chains, or plugin-exposed metadata across the stack.
---

# Plugin End to End

## Trigger

- Add or change a plugin type
- Change plugin config schema, execution semantics, or plugin-visible headers
- Update plugin chain behavior that affects runtime or tests

## Required Surfaces

- `router_config_contract`
- `router_runtime`
- `python_cli_schema`
- `local_e2e`
- `ci_e2e`

## Conditional Surfaces

- `response_headers`
- `dashboard_config_ui`
- `topology_visualization`
- `playground_reveal`
- `docs_examples`

## Stop Conditions

- Plugin behavior is ambiguous between config, runtime, and user-visible surfaces

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/playbooks/vllm-sr-cli-docker.md](../../../../docs/agent/playbooks/vllm-sr-cli-docker.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Plugin config and runtime behavior stay aligned
- Tests and E2E cover the changed plugin path
- User-visible plugin metadata is updated wherever it is displayed
