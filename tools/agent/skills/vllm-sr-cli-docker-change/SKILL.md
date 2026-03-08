---
name: vllm-sr-cli-docker-change
category: legacy_reference
description: Legacy CLI/startup reference skill retained while project-level primary skills become the default entry point.
---

# Legacy CLI And Docker Change

## Trigger

- Use this only as a reference after a project-level primary skill is selected and the change is centered on CLI/startup implementation details

## Must Read

- Read [docs/agent/playbooks/vllm-sr-cli-docker.md](../../../../docs/agent/playbooks/vllm-sr-cli-docker.md)
- Prefer the `startup-chain-change` primary skill or `python-cli-schema` fragment for current guidance

## Standard Commands

- `make agent-report ENV=cpu|amd CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu|amd CHANGED_FILES="..."`

## Acceptance

- The active primary skill remains authoritative and this file only points to the CLI/startup playbook
