---
name: signal-end-to-end
category: primary
description: Use when adding or changing a signal that can span router config, runtime evaluation, native bindings, Python CLI schema, dashboard surfaces, headers, topology/playground, and E2E.
---

# Signal End to End

## Trigger

- Add a new signal type or signal rule shape
- Change how an existing signal is configured, matched, emitted, or displayed
- Touch router signal config plus Python CLI schema in the same feature

## Required Surfaces

- `router_config_contract`
- `router_runtime`
- `python_cli_schema`
- `local_e2e`
- `ci_e2e`

## Conditional Surfaces

- `native_binding`
- `response_headers`
- `dashboard_config_ui`
- `topology_visualization`
- `playground_reveal`
- `dsl_crd`
- `docs_examples`

## Stop Conditions

- The signal contract is unclear across router config, CLI schema, and dashboard serialization
- The feature needs a new native runtime path that cannot be built or validated here

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/playbooks/go-router.md](../../../../docs/agent/playbooks/go-router.md)
- [docs/agent/playbooks/vllm-sr-cli-docker.md](../../../../docs/agent/playbooks/vllm-sr-cli-docker.md)
- [docs/agent/feature-complete-checklist.md](../../../../docs/agent/feature-complete-checklist.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Router config/runtime and Python CLI schema agree on the signal contract
- Any user-visible signal metadata updates the header/UI surfaces that expose it
- Relevant E2E coverage is added or updated when behavior changes

## Example

- For a new signal that also exposes a header:
  - update router config and runtime matching
  - update Python CLI models/parser/merger
  - update binding code only if the signal needs new native inference
  - update `HeaderDisplay`, `HeaderReveal`, `ChatComponent`, config UI, topology/playground only if the signal becomes user-visible
