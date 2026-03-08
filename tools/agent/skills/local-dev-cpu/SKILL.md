---
name: local-dev-cpu
category: fragment
description: Canonical CPU-local image build, serve, and smoke workflow.
---

# Local Dev CPU

## Trigger

- The primary skill needs `cpu-local` build/serve/smoke validation

## Required Surfaces

- `local_smoke`

## Conditional Surfaces

- `local_e2e`

## Stop Conditions

- CPU-local smoke cannot be run in the current workspace/runtime

## Must Read

- [docs/agent/environments.md](../../../../docs/agent/environments.md)
- [docs/agent/playbooks/vllm-sr-cli-docker.md](../../../../docs/agent/playbooks/vllm-sr-cli-docker.md)

## Standard Commands

- `make agent-dev ENV=cpu`
- `make agent-serve-local ENV=cpu`
- `make agent-smoke-local`

## Acceptance

- The default CPU smoke config starts successfully
- `vllm-sr status all` and dashboard smoke checks pass
