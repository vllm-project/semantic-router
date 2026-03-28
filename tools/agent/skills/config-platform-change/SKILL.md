---
name: config-platform-change
category: primary
description: Synchronizes config representations across router config, Python CLI schema, and dashboard config UI. Use when adding or changing a config concept that spans those surfaces or addressing config representation debt before Kubernetes-facing translation.
---

# Config Platform Change

## Trigger

- Change a config concept that exists in router config and Python CLI schema
- Change config translation between router config and dashboard config UI
- Work on config complexity or representation debt before Kubernetes-facing translation

## Workflow

1. Read change surfaces and module boundaries to identify all config layers affected
2. Modify the config concept across all touched surfaces (router, CLI, dashboard)
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to verify surface consistency
4. Run `make agent-ci-gate CHANGED_FILES="..."` to validate constraints
5. Record any intentional remaining mismatches as indexed debt entries

## Gotchas

- Do not update the router config contract without also checking CLI and dashboard config serialization in the same change.
- If defaults, field names, or compatibility policy change, make the migration posture explicit or record the intentional gap as indexed debt.

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- The same config concept is represented consistently across router config, CLI schema, and dashboard config UI
- Any intentional remaining mismatch is recorded in the matching indexed debt entry in the same change
- Platform-facing translation tests or validations are updated when behavior changes
