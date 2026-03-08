---
name: config-platform-change
category: primary
description: Use when a config representation must stay aligned across router config, CLI schema, dashboard config UI, and platform translation layers.
---

# Config Platform Change

## Trigger

- Change a config concept that exists in router config and Python CLI schema
- Change config translation between router config, dashboard config UI, DSL, or Kubernetes forms
- Work on config complexity or representation debt across deployment surfaces

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/tech-debt-register.md](../../../../docs/agent/tech-debt-register.md)
- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- The same config concept is represented consistently across the touched config surfaces
- Any intentional remaining mismatch is recorded in the tech-debt register in the same change
- Platform-facing translation tests or validations are updated when behavior changes
