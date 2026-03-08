---
name: dashboard-surface-change
category: primary
description: Use when changing frontend dashboard config, topology, or playground surfaces that reflect router behavior.
---

# Dashboard Surface Change

## Trigger

- Change config editing UI, topology visualization, or playground reveal or display
- Change frontend handling of routing metadata or router-backed config

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/feature-complete-checklist.md](../../../../docs/agent/feature-complete-checklist.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`

## Acceptance

- Frontend dashboard surfaces remain aligned with router and CLI contracts
- User-visible routing metadata stays consistent across config, topology, and playground
