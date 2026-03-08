---
name: dashboard-console-platform-change
category: primary
description: Use when changing dashboard backend, console persistence, auth, or control-plane behavior behind the dashboard surface.
---

# Dashboard Console Platform Change

## Trigger

- Change dashboard backend APIs, persistence, or console-side server behavior
- Change dashboard auth, session, or storage behavior
- Work on dashboard enterprise-console platform debt

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/tech-debt-register.md](../../../../docs/agent/tech-debt-register.md)
- [docs/agent/feature-complete-checklist.md](../../../../docs/agent/feature-complete-checklist.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Console backend changes remain aligned with frontend callers and local smoke expectations
- Any new auth, storage, or session behavior is documented in the canonical harness or debt register
