---
name: local-dev-amd
category: fragment
description: Canonical AMD-local image build, serve, and smoke workflow.
---

# Local Dev AMD

## Trigger

- The primary skill needs `amd-local` build/serve/smoke validation

## Required Surfaces

- `local_smoke`

## Conditional Surfaces

- `local_e2e`

## Stop Conditions

- AMD-local smoke cannot be run or platform image mapping is unavailable

## Must Read

- [docs/agent/amd-local.md](../../../../docs/agent/amd-local.md)
- [docs/agent/environments.md](../../../../docs/agent/environments.md)
- [deploy/amd/README.md](../../../../deploy/amd/README.md)
- [deploy/amd/config.yaml](../../../../deploy/amd/config.yaml)

## Standard Commands

- `make agent-dev ENV=amd`
- `make agent-serve-local ENV=amd`
- `make agent-smoke-local`
- `make agent-serve-local ENV=amd AGENT_SERVE_CONFIG=deploy/amd/config.yaml`

## Acceptance

- The default AMD smoke config starts successfully
- AMD image/platform behavior does not fall back unexpectedly
- When real AMD model deployment is in scope, the agent uses `deploy/amd/README.md` and `deploy/amd/config.yaml` as the primary reference instead of inventing a new ROCm setup path
