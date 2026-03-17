---
name: startup-chain-change
category: primary
description: Modifies the local startup chain including image build, container serve/bootstrap logic, and canonical smoke test behavior. Use when changing `vllm-sr serve` behavior, image selection or pull policy, container startup sequences, local Docker/Make bootstrap, or canonical smoke config.
---

# Startup Chain Change

## Trigger

- Change `vllm-sr serve`, image selection, pull policy, or container startup behavior
- Change canonical local smoke config or agent smoke flow
- Change local Docker or Make bootstrap behavior

## Workflow

1. Read environment docs and CLI Docker playbook for startup chain context
2. Modify serve, bootstrap, image selection, or smoke config
3. Run `make agent-report ENV=cpu|amd CHANGED_FILES="..."` to identify impacted surfaces
4. Run `make agent-feature-gate ENV=cpu|amd CHANGED_FILES="..."` to validate startup behavior
5. Verify the canonical local serve path works with the default smoke config

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
