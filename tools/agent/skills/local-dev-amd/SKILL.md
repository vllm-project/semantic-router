---
name: local-dev-amd
category: fragment
description: Builds Docker images, starts local servers, and runs smoke tests for the AMD/ROCm development environment. Use when validating changes locally on AMD hardware, building AMD container images, or running AMD-specific smoke and E2E tests.
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

## Workflow

1. Read AMD-local docs and environment config to understand the AMD build setup
2. Build the AMD image with `make agent-dev ENV=amd`
3. Start the local server with `make agent-serve-local ENV=amd`
4. Run smoke tests with `make agent-smoke-local` to validate the build
5. Verify the default AMD smoke config starts successfully without unexpected fallbacks

## Must Read

- [docs/agent/amd-local.md](../../../../docs/agent/amd-local.md)
- [docs/agent/environments.md](../../../../docs/agent/environments.md)
- [deploy/amd/README.md](../../../../deploy/amd/README.md)
- [deploy/recipes/balance.yaml](../../../../deploy/recipes/balance.yaml)

## Standard Commands

- `make agent-dev ENV=amd`
- `make agent-serve-local ENV=amd`
- `make agent-smoke-local`
- `make agent-serve-local ENV=amd AGENT_SERVE_CONFIG=deploy/recipes/balance.yaml`

## Acceptance

- The default AMD smoke config starts successfully
- AMD image/platform behavior does not fall back unexpectedly
- When real AMD model deployment is in scope, the agent uses `deploy/amd/README.md` and `deploy/recipes/balance.yaml` as the primary reference instead of inventing a new ROCm setup path
