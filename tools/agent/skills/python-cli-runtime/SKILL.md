---
name: python-cli-runtime
category: fragment
description: Python CLI command orchestration, local image management, and serve or status runtime behavior.
---

# Python CLI Runtime

## Trigger

- The primary skill touches CLI startup, serve, bootstrap, status, or local image behavior

## Must Read

- [docs/agent/environments.md](../../../../docs/agent/environments.md)
- [docs/agent/playbooks/vllm-sr-cli-docker.md](../../../../docs/agent/playbooks/vllm-sr-cli-docker.md)
- [../../../../src/vllm-sr/cli/AGENTS.md](../../../../src/vllm-sr/cli/AGENTS.md)

## Standard Commands

- `make agent-dev ENV=cpu|amd`
- `make agent-serve-local ENV=cpu|amd`
- `make vllm-sr-test`

## Acceptance

- CLI runtime orchestration and local smoke behavior stay aligned
