# TD009: E2E Profile Inventory, Naming, and Documentation Have Drifted Out of Sync

## Status

Open

## Scope

test discoverability and profile inventory

## Summary

The documented supported profile list, the actual runner inventory, and the harness profile map no longer present one synchronized view of the active profile set.

## Evidence

- [e2e/README.md](../../../e2e/README.md)
- [e2e/cmd/e2e/main.go](../../../e2e/cmd/e2e/main.go)
- [tools/agent/e2e-profile-map.yaml](../../../tools/agent/e2e-profile-map.yaml)
- [e2e/profiles/response-api/profile.go](../../../e2e/profiles/response-api/profile.go)
- [e2e/profiles/responseapi/redis_profile.go](../../../e2e/profiles/responseapi/redis_profile.go)

## Why It Matters

- The documented supported profile list in `e2e/README.md` does not match the actual runner inventory in `e2e/cmd/e2e/main.go`, and the harness profile map has a different set again.
- The same surface uses mixed naming conventions such as `response-api` for a public profile directory and `responseapi` for shared helper code, while some profiles are present in the runner but absent from the harness map.
- Inventory drift makes it harder to discover supported profiles, reason about affected coverage, and reliably wire new profiles into docs, CI, and agent selection.

## Desired End State

- One authoritative inventory of supported profiles, their ownership, and their classification in local versus CI coverage.
- Naming between profile directories, helper packages, docs, and harness manifests follows a consistent convention.

## Exit Criteria

- `e2e/README.md`, `e2e/cmd/e2e/main.go`, and `tools/agent/e2e-profile-map.yaml` stay synchronized for the active profile set.
- Profile additions or removals can be mechanically validated against the canonical inventory.
