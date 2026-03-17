---
name: config-platform-change
category: primary
description: Synchronizes config representations across router config, Python CLI schema, dashboard config UI, DSL, and Kubernetes translation layers. Use when adding or changing a config concept that spans multiple surfaces, updating config translation logic, or addressing config representation debt across deployment targets.
---

# Config Platform Change

## Trigger

- Change a config concept that exists in router config and Python CLI schema
- Change config translation between router config, dashboard config UI, DSL, or Kubernetes forms
- Work on config complexity or representation debt across deployment surfaces

## Workflow

1. Read change surfaces and module boundaries to identify all config layers affected
2. Modify the config concept across all touched surfaces (router, CLI, dashboard, DSL, k8s)
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to verify surface consistency
4. Run `make agent-ci-gate CHANGED_FILES="..."` to validate constraints
5. Record any intentional remaining mismatches as indexed debt entries

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/tech-debt-register.md](../../../../docs/agent/tech-debt-register.md)
- [docs/agent/tech-debt/README.md](../../../../docs/agent/tech-debt/README.md)
- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- The same config concept is represented consistently across the touched config surfaces
- Any intentional remaining mismatch is recorded in the matching indexed debt entry in the same change
- Platform-facing translation tests or validations are updated when behavior changes
