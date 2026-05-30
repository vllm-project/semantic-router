# TD034: Runtime and Dashboard State Surfaces Still Lack a Coherent Durability, Recovery, and Telemetry Contract

## Status

Open

## Owner Plan

PL0033 v0.3 Themis Release Closure

## Release Relevance

v0.3 Themis

## Scope

Router runtime state, dashboard workflow/auth/evaluation state, CLI local
runtime mounts, model-selection learning state, and the docs/defaults that tell
operators what survives restart.

## Summary

The repo exposes several stateful surfaces beyond request routing: semantic
cache, response storage, replay, vector metadata, startup status, model-selection
learning state, dashboard jobs, OpenClaw rooms/messages, auth sessions,
browser-local playground state, generated config, and local runtime files. Some
of these are now backed by Redis, Postgres, SQLite, or runtime registries. Others
still rely on memory, local files, browser storage, or process-wide fallback
paths.

The remaining debt is to turn this into one readable product contract: which
state is ephemeral, which survives restart, which requires shared durable
storage, and which telemetry/progress records can be used for recovery.

## Evidence

- [docs/architecture/state-taxonomy-and-inventory.md](../../architecture/state-taxonomy-and-inventory.md)
- [src/semantic-router/pkg/config/](../../../src/semantic-router/pkg/config)
- [src/semantic-router/pkg/responsestore/](../../../src/semantic-router/pkg/responsestore)
- [src/semantic-router/pkg/routerreplay/](../../../src/semantic-router/pkg/routerreplay)
- [src/semantic-router/pkg/vectorstore/](../../../src/semantic-router/pkg/vectorstore)
- [src/semantic-router/pkg/startupstatus/](../../../src/semantic-router/pkg/startupstatus)
- [src/semantic-router/pkg/selection/](../../../src/semantic-router/pkg/selection)
- [dashboard/backend/workflowstore/](../../../dashboard/backend/workflowstore)
- [dashboard/backend/auth/](../../../dashboard/backend/auth)
- [dashboard/backend/evaluation/](../../../dashboard/backend/evaluation)
- [dashboard/backend/mlpipeline/](../../../dashboard/backend/mlpipeline)
- [dashboard/frontend/src/hooks/useConversationStorage.ts](../../../dashboard/frontend/src/hooks/useConversationStorage.ts)
- [dashboard/frontend/src/utils/authFetch.ts](../../../dashboard/frontend/src/utils/authFetch.ts)
- [src/vllm-sr/cli/](../../../src/vllm-sr/cli)

## Why It Matters

- Operators need to know what survives restart before trusting dashboard,
  response, replay, vector, auth, and model-selection features.
- Recovery behavior is fragile when each surface chooses memory, file, DB,
  browser, or runtime-registry storage independently.
- Maintainers cannot reliably automate release readiness without one state
  inventory tied to tests and docs.

## Desired End State

- Each state surface has an explicit durability class, storage owner, recovery
  expectation, and test/doc reference.
- Runtime and dashboard state that must survive restart uses a supported durable
  backend or clearly declares itself local/ephemeral.
- Browser-local state is bounded and treated as UX cache, not product truth.
- Telemetry and progress records that drive recovery are durable enough for the
  workflows that consume them.

## Exit Criteria

- The state taxonomy matches current source and release docs.
- v0.3 restart/recovery checks cover the release-critical state surfaces.
- Dashboard auth/session, workflow, OpenClaw, response/replay/vector, and
  model-selection state each have a clear storage and recovery contract.
- No release-critical path depends on ambiguous process-wide or browser-local
  state for steady-state behavior.
