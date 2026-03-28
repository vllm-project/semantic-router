---
name: python-cli-runtime
category: fragment
description: Manages Python CLI command orchestration including startup, serve, bootstrap, status checks, and local Docker image management. Use when modifying CLI startup sequences, changing how local images are built or managed, updating serve behavior, or adjusting status command output.
---

# Python CLI Runtime

## Trigger

- The primary skill touches CLI startup, serve, bootstrap, status, or local image behavior

## Workflow

1. Read environment docs, CLI Docker playbook, and CLI AGENTS.md for runtime context
2. Modify CLI startup, serve, bootstrap, status, or local image behavior
3. Run `make vllm-sr-test` to validate CLI unit behavior
4. Run `make agent-serve-local ENV=cpu|amd` to verify local smoke behavior
5. Confirm CLI runtime orchestration and local smoke behavior stay aligned

## Must Read

- [docs/agent/playbooks/vllm-sr-cli-docker.md](../../../../docs/agent/playbooks/vllm-sr-cli-docker.md)
- [../../../../src/vllm-sr/cli/AGENTS.md](../../../../src/vllm-sr/cli/AGENTS.md)

## Standard Commands

- `make agent-dev ENV=cpu|amd`
- `make agent-serve-local ENV=cpu|amd`
- `make vllm-sr-test`

## Acceptance

- CLI runtime orchestration and local smoke behavior stay aligned
