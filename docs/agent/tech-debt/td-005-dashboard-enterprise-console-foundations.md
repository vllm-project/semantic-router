# TD005: Dashboard Lacks Enterprise Console Foundations

## Status

Open

## Scope

dashboard product architecture

## Summary

The dashboard already provides setup, deploy, proxy, and readonly flows, but it does not yet provide the persistence, authentication, and session controls expected from an enterprise console.

## Evidence

- [dashboard/README.md](../../../dashboard/README.md)
- [dashboard/backend/config/config.go](../../../dashboard/backend/config/config.go)
- [dashboard/backend/evaluation/db.go](../../../dashboard/backend/evaluation/db.go)
- [dashboard/backend/router/router.go](../../../dashboard/backend/router/router.go)

## Why It Matters

- The dashboard already provides readonly mode, proxying, setup/deploy flows, and a small evaluation database, but it does not yet provide a unified persistent config store, user login/session management, or stronger enterprise security controls.
- The README explicitly treats OIDC, RBAC, and stronger proxy/session behavior as future work.
- This limits the dashboard's role as a real enterprise console.

## Desired End State

- Dashboard state and config persistence move toward a clearer control-plane model.
- Authentication, authorization, and user/session management become first-class capabilities instead of future notes.

## Exit Criteria

- The dashboard has a coherent persistent storage model for console state and config workflows.
- Auth, login/session, and user/role controls exist as supported product features rather than roadmap notes.
