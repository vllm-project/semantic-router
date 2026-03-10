---
name: plugin-end-to-end
category: primary
description: Use when changing plugin config or plugin runtime behavior that spans router config, post-decision processing, optional CLI/UI exposure, and E2E.
---

# Plugin End to End

## Trigger

- Add or change a plugin type
- Change plugin config schema, execution semantics, or plugin-exposed metadata
- Update plugin chain behavior that affects runtime or tests

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)
- [docs/agent/playbooks/go-router.md](../../../../docs/agent/playbooks/go-router.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Plugin config and post-decision runtime behavior stay aligned
- Tests and E2E cover the changed plugin path
- User-visible plugin metadata is updated wherever it is displayed
