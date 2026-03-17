---
name: dashboard-console-platform-change
category: primary
description: Modifies dashboard backend APIs, console persistence, authentication, session management, and control-plane behavior behind the dashboard surface. Use when changing server-side dashboard handlers, updating auth/session logic, modifying storage behavior, or addressing enterprise-console platform debt.
---

# Dashboard Console Platform Change

## Trigger

- Change dashboard backend APIs, persistence, or console-side server behavior
- Change dashboard auth, session, or storage behavior
- Work on dashboard enterprise-console platform debt

## Workflow

1. Read change surfaces and tech debt docs to understand dashboard platform dependencies
2. Modify backend APIs, persistence, auth, or session logic
3. Run `make agent-report ENV=cpu CHANGED_FILES="..."` to identify impacted surfaces
4. Run `make agent-ci-gate CHANGED_FILES="..."` to validate constraints
5. Document any new auth, storage, or session behavior in the canonical harness or create a debt entry

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/tech-debt-register.md](../../../../docs/agent/tech-debt-register.md)
- [docs/agent/tech-debt/README.md](../../../../docs/agent/tech-debt/README.md)
- [docs/agent/feature-complete-checklist.md](../../../../docs/agent/feature-complete-checklist.md)

## Standard Commands

- `make agent-report ENV=cpu CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."`

## Acceptance

- Console backend changes remain aligned with frontend callers and local smoke expectations
- Any new auth, storage, or session behavior is documented in the canonical harness or tracked through the matching indexed debt entry
