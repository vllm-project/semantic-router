---
name: local-dev-cpu
category: fragment
description: Builds Docker images, starts local servers, and runs smoke tests for the CPU-only development environment. Use when validating changes locally on CPU, building CPU container images, or running CPU-specific smoke and E2E tests.
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

## Workflow

1. Read environment docs and CLI Docker playbook to understand the CPU build setup
2. Build the CPU image with `make agent-dev ENV=cpu`
3. Start the local server with `make agent-serve-local ENV=cpu`
4. Run smoke tests with `make agent-smoke-local` to validate the build
5. Verify `vllm-sr status all` and dashboard smoke checks pass

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
