---
name: plugin-end-to-end
category: primary
description: Implements end-to-end plugin changes spanning router config, post-decision processing, optional CLI/UI exposure, and E2E test coverage. Use when adding a new plugin type, changing plugin config schema or execution semantics, updating plugin chain behavior, or modifying plugin-exposed metadata across surfaces.
---

# Plugin End to End

## Trigger

- Add or change a plugin type
- Change plugin config schema, execution semantics, or plugin-exposed metadata
- Update plugin chain behavior that affects runtime or tests

## Workflow

1. Read change surfaces, module boundaries, and Go router playbook for plugin context
2. Modify plugin config schema, execution semantics, or chain behavior
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to identify impacted surfaces
4. Run `make agent-ci-gate CHANGED_FILES="..."` to validate all constraints
5. Verify plugin config, runtime behavior, tests, and E2E cover the changed plugin path

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
