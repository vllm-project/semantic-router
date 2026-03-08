---
name: python-cli-schema
category: fragment
description: Python CLI schema, parser, validator, merger, and config-translation details.
---

# Python CLI Schema

## Trigger

- The primary skill touches `src/vllm-sr/cli` schema or translation code

## Must Read

- [docs/agent/playbooks/vllm-sr-cli-docker.md](../../../../docs/agent/playbooks/vllm-sr-cli-docker.md)

## Standard Commands

- `make vllm-sr-test`
- `make vllm-sr-test-integration`

## Acceptance

- CLI schema, parser, validator, and merger express the same contract as the router config they translate
