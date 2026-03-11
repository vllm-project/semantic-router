---
name: startup-chain-change
category: primary
description: Use when changing local image build, serve or bootstrap logic, or canonical smoke behavior.
---

# Startup Chain Change

## Trigger

- Change `vllm-sr serve`, image selection, pull policy, or container startup behavior
- Change canonical local smoke config or agent smoke flow
- Change local Docker or Make bootstrap behavior

## Must Read

- [docs/agent/environments.md](../../../../docs/agent/environments.md)
- [docs/agent/amd-local.md](../../../../docs/agent/amd-local.md)
- [docs/agent/playbooks/vllm-sr-cli-docker.md](../../../../docs/agent/playbooks/vllm-sr-cli-docker.md)

## Standard Commands

- `make agent-report ENV=cpu|amd CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu|amd CHANGED_FILES="..."`

## Acceptance

- The canonical local serve path works with the default smoke config
- Startup-chain changes include local smoke plus relevant CLI or integration coverage
