---
name: startup-chain-change
category: primary
description: Use when changing local image build, serve/bootstrap logic, canonical smoke behavior, or startup-chain wiring.
---

# Startup Chain Change

## Trigger

- Change `vllm-sr serve`, image selection, pull policy, or container startup behavior
- Change canonical local smoke config or agent smoke flow
- Change local Docker/Make bootstrap behavior

## Required Surfaces

- `local_smoke`
- `python_cli_schema`

## Conditional Surfaces

- `response_headers`
- `local_e2e`
- `ci_e2e`
- `docs_examples`

## Stop Conditions

- The startup path cannot be validated locally in the target environment

## Must Read

- [docs/agent/environments.md](../../../../docs/agent/environments.md)
- [docs/agent/playbooks/vllm-sr-cli-docker.md](../../../../docs/agent/playbooks/vllm-sr-cli-docker.md)

## Standard Commands

- `make agent-report ENV=cpu|amd CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu|amd CHANGED_FILES="..."`

## Acceptance

- The canonical local serve path works with the default smoke config
- Startup-chain changes include local smoke plus relevant CLI/integration coverage
