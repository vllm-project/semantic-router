---
name: dashboard-surface-change
category: primary
description: Modifies frontend dashboard surfaces including config editing UI, topology visualization, and playground reveal/display components that reflect router behavior. Use when changing how routing metadata is presented in the dashboard, updating config editing forms, modifying topology graph rendering, or adjusting playground response display.
---

# Dashboard Surface Change

## Trigger

- Change config editing UI, topology visualization, or playground reveal or display
- Change frontend handling of routing metadata or router-backed config

## Workflow

1. Read change surfaces and feature-complete checklist to understand dashboard dependencies
2. Modify config editing UI, topology visualization, or playground display components
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to identify impacted surfaces
4. Run `make agent-ci-gate CHANGED_FILES="..."` to validate alignment with router and CLI contracts

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/feature-complete-checklist.md](../../../../docs/agent/feature-complete-checklist.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`

## Acceptance

- Frontend dashboard surfaces remain aligned with router and CLI contracts
- User-visible routing metadata stays consistent across config, topology, and playground
