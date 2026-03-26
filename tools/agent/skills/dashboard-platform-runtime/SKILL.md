---
name: dashboard-platform-runtime
category: fragment
description: Maintains dashboard frontend and backend implementation slices that present, configure, or expose router behavior through the console. Use when the task changes dashboard config flows, console backend handlers, topology rendering, playground reveal flows, or other user-visible routing metadata in the dashboard.
---

# Dashboard Platform Runtime

## Trigger

- The primary skill touches dashboard config flows, console backend handlers, topology rendering, playground reveal, or other user-visible routing metadata in the dashboard

## Workflow

1. Read change surfaces and the nearest dashboard backend rules to understand the affected console path
2. Modify the frontend or backend dashboard slice without splitting one console contract across multiple fragment mental models
3. Run `make dashboard-check` to validate dashboard consistency
4. Verify config flows, topology or reveal metadata, backend callers, and user-visible keys stay aligned

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [../../../../dashboard/backend/handlers/AGENTS.md](../../../../dashboard/backend/handlers/AGENTS.md)

## Standard Commands

- `make dashboard-check`

## Acceptance

- Dashboard config flows, topology or reveal metadata, and backend callers stay aligned on the same console contract
