# TD005: Dashboard Lacks Enterprise Console Foundations

## Status

Open

## Scope

dashboard product architecture

## Summary

The dashboard already provides setup, deploy, proxy, readonly, and basic login flows, but it still does not provide the cookie-backed session and server-enforced browser access controls expected from an enterprise console.

## Evidence

- [dashboard/README.md](../../../dashboard/README.md)
- [dashboard/backend/config/config.go](../../../dashboard/backend/config/config.go)
- [dashboard/backend/evaluation/db.go](../../../dashboard/backend/evaluation/db.go)
- [dashboard/backend/router/router.go](../../../dashboard/backend/router/router.go)
- [dashboard/backend/auth/middleware.go](../../../dashboard/backend/auth/middleware.go)
- [dashboard/frontend/src/utils/authFetch.ts](../../../dashboard/frontend/src/utils/authFetch.ts)

## Why It Matters

- The dashboard already provides readonly mode, proxying, setup/deploy flows, a small evaluation database, and bearer-token login, but it does not yet provide a unified persistent config store, cookie-backed browser sessions, or stronger enterprise security controls.
- Browser-only bearer tokens still force temporary query-token fallbacks for iframe, EventSource, and WebSocket console surfaces until a stronger session model exists.
- The README explicitly treats OIDC, RBAC, and stronger proxy/session behavior as future work.
- This limits the dashboard's role as a real enterprise console.

## Desired End State

- Dashboard state and config persistence move toward a clearer control-plane model.
- Authentication, authorization, and user/session management become first-class capabilities instead of future notes.
- Browser-facing routes and embedded console surfaces use a server-enforced session model instead of localStorage bearer tokens and query-token fallbacks.

## Exit Criteria

- The dashboard has a coherent persistent storage model for console state and config workflows.
- Auth, login/session, and user/role controls exist as supported product features rather than roadmap notes.
- Internal HTML, iframe, SSE, and WebSocket console surfaces no longer rely on bearer tokens in localStorage or URL query parameters for runtime access control.
