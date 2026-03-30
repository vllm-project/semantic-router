---
name: dashboard-platform-change
category: primary
description: Modifies dashboard frontend or backend surfaces that present, configure, or manage router behavior through the console UI. Use when changing dashboard pages or components, backend handlers, console persistence, auth or session flows, or user-visible routing metadata in the dashboard.
---

# Dashboard Platform Change

## Trigger

- Change dashboard frontend pages, components, or config editing flows
- Change dashboard backend handlers, console persistence, auth, or session behavior
- Change dashboard-only reveal, display, or topology presentation when the upstream router contract is already stable

## Workflow

1. Read change surfaces and the nearest dashboard rules to understand the affected console flow
2. Modify the dashboard frontend or backend surface and identify whether the change is config UI, routing visibility, or console-platform behavior
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to identify impacted surfaces
4. Run `make agent-ci-gate CHANGED_FILES="..."` to validate the affected constraints
5. Verify frontend callers, backend handlers, and user-visible routing metadata stay aligned

## Gotchas

- Dashboard-only presentation work still encodes upstream contracts; verify the emitted data shape before treating the change as cosmetic.
- Backend console work can still break frontend callers or local smoke, so validate both ends of the console path.
- Keep router-runtime semantics upstream; if the change originates in signal or plugin behavior, let the router-side primary own it and treat dashboard updates as dependent surfaces.

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/module-boundaries.md](../../../../docs/agent/module-boundaries.md)
- [../../../../dashboard/backend/handlers/AGENTS.md](../../../../dashboard/backend/handlers/AGENTS.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Dashboard frontend and backend surfaces remain aligned with router and CLI contracts
- User-visible routing metadata stays consistent across config, topology, playground, and console callers
- Any new auth, storage, or session behavior is documented in the canonical harness or tracked through the matching indexed debt entry
