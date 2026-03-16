---
name: dashboard-console-backend
category: fragment
description: Modifies dashboard backend handlers, persistence layer, authentication, session management, and server-side console behavior. Use when changing dashboard API endpoints, updating database queries, modifying auth/session logic, or adjusting server-side console functionality.
---

# Dashboard Console Backend

## Trigger

- The primary skill touches dashboard backend handlers, persistence, auth, or session behavior

## Workflow

1. Read change surfaces and tech debt docs to understand backend dependencies
2. Modify backend handlers, persistence, auth, or session behavior
3. Run `make dashboard-check` to validate UI and backend consistency
4. Verify backend console handlers, persistence, and callers stay aligned

## Must Read

- [docs/agent/change-surfaces.md](../../../../docs/agent/change-surfaces.md)
- [docs/agent/tech-debt-register.md](../../../../docs/agent/tech-debt-register.md)
- [docs/agent/tech-debt/README.md](../../../../docs/agent/tech-debt/README.md)

## Standard Commands

- `make dashboard-check`

## Acceptance

- Backend console handlers, persistence, and callers stay aligned
