# TD009: E2E Profile Inventory, Naming, and Documentation Have Drifted Out of Sync

## Status

Closed

## Scope

test discoverability and profile inventory

## Summary

The documented supported profile list, the actual runner inventory, and the harness profile map previously drifted out of sync. The repo now has one mechanically checked runnable profile inventory and consistent response-api helper naming, so this gap is retired.

## Evidence

- [e2e/README.md](../../../e2e/README.md)
- [e2e/profiles/all/imports.go](../../../e2e/profiles/all/imports.go)
- [e2e/pkg/framework/profile_registry.go](../../../e2e/pkg/framework/profile_registry.go)
- [tools/agent/e2e-profile-map.yaml](../../../tools/agent/e2e-profile-map.yaml)
- [tools/agent/scripts/agent_ci_validation.py](../../../tools/agent/scripts/agent_ci_validation.py)
- [e2e/profiles/response-api/profile.go](../../../e2e/profiles/response-api/profile.go)
- [e2e/profiles/response-api-shared/redis_profile.go](../../../e2e/profiles/response-api-shared/redis_profile.go)

## Why It Matters

- The documented supported profile list in `e2e/README.md` does not match the actual runner inventory in `e2e/cmd/e2e/main.go`, and the harness profile map has a different set again.
- The same surface uses mixed naming conventions such as `response-api` for a public profile directory and `responseapi` for shared helper code, while some profiles are present in the runner but absent from the harness map.
- Inventory drift makes it harder to discover supported profiles, reason about affected coverage, and reliably wire new profiles into docs, CI, and agent selection.

## Desired End State

- One authoritative inventory of supported profiles, their ownership, and their classification in local versus CI coverage.
- Naming between profile directories, helper packages, docs, and harness manifests follows a consistent convention.

## Exit Criteria

- Satisfied on 2026-03-08: `e2e/README.md`, `e2e/profiles/all/imports.go`, and `tools/agent/e2e-profile-map.yaml` now stay synchronized for the active profile set.
- Satisfied on 2026-03-08: profile additions or removals can be mechanically validated against the canonical inventory, and the last `responseapi` helper naming drift is gone.

## Resolution

- The runnable profile inventory is now mechanically synchronized across `e2e/README.md`, `e2e/profiles/all/imports.go`, and `tools/agent/e2e-profile-map.yaml`.
- `tools/agent/scripts/agent_ci_validation.py` now checks that runnable inventory drift before the harness accepts a change.
- The shared Redis-backed helper moved from `e2e/profiles/responseapi/` to `e2e/profiles/response-api-shared/`, so the response-api E2E stack no longer exposes a divergent helper naming convention.
