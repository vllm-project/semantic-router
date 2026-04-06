# TD005: Dashboard Lacks Enterprise Console Foundations

## Status

Closed

## Scope

dashboard product architecture

## Summary

The dashboard now provides the enterprise-console foundation this debt was missing. Browser-facing auth uses a server-set `HttpOnly` session cookie instead of frontend-owned bearer-token storage, embedded console surfaces no longer depend on `authToken` query parameters, and the state taxonomy explicitly classifies browser-local chat history as convenience state instead of leaving it as an implicit product contract.

## Evidence

- [dashboard/README.md](../../../dashboard/README.md)
- [dashboard/backend/config/config.go](../../../dashboard/backend/config/config.go)
- [dashboard/backend/evaluation/db.go](../../../dashboard/backend/evaluation/db.go)
- [dashboard/backend/router/router.go](../../../dashboard/backend/router/router.go)
- [dashboard/backend/auth/http_handlers.go](../../../dashboard/backend/auth/http_handlers.go)
- [dashboard/backend/auth/middleware.go](../../../dashboard/backend/auth/middleware.go)
- [dashboard/backend/auth/session_cookie.go](../../../dashboard/backend/auth/session_cookie.go)
- [dashboard/backend/auth/handlers_test.go](../../../dashboard/backend/auth/handlers_test.go)
- [dashboard/frontend/src/utils/authFetch.ts](../../../dashboard/frontend/src/utils/authFetch.ts)
- [dashboard/frontend/src/contexts/AuthContext.tsx](../../../dashboard/frontend/src/contexts/AuthContext.tsx)
- [dashboard/frontend/e2e/auth-flow.spec.ts](../../../dashboard/frontend/e2e/auth-flow.spec.ts)
- [docs/agent/state-taxonomy-and-inventory.md](../state-taxonomy-and-inventory.md)

## Why It Matters

- The dashboard already provided readonly mode, proxying, setup/deploy flows, a small evaluation database, and user or role primitives, but it still exposed browser auth through frontend-owned bearer-token storage and URL token transport.
- Query-token fallbacks across iframe, EventSource, and WebSocket surfaces weakened the security posture of embedded console pages and made session revocation or inspection harder to reason about.
- Until the repo explicitly classified browser-local chat history, contributors still had to guess whether localStorage-backed conversation state was demo-only convenience state or an unfinished shared workflow contract.
- This limited the dashboard's role as a real enterprise console.

## Desired End State

- Dashboard state and config persistence move toward a clearer control-plane model.
- Authentication, authorization, and user/session management become first-class capabilities instead of future notes.
- Browser-facing routes and embedded console surfaces use a server-enforced session model instead of localStorage bearer tokens and query-token fallbacks.

## Exit Criteria

- Satisfied on 2026-04-06: the dashboard has a coherent persistent storage model for console state and config workflows through the state taxonomy, active-config projection, SQLite-backed auth store, and explicit browser-local convenience-state classifications.
- Satisfied on 2026-04-06: auth, login/session, and user/role controls exist as supported product features rather than roadmap notes, and the dashboard README now documents cookie-backed browser sessions as the current contract.
- Satisfied on 2026-04-06: internal HTML, iframe, SSE, and WebSocket console surfaces no longer rely on bearer tokens in `localStorage` or URL query parameters for runtime access control.

## Retirement Notes

- `dashboard/backend/auth/http_handlers.go` now sets a same-origin `HttpOnly` `vsr_session` cookie on login and first-admin bootstrap, while `logout` clears that cookie server-side instead of expecting the browser to manage session state itself.
- `dashboard/backend/auth/middleware.go` still accepts `Authorization: Bearer` for API-style callers, but browser requests now resolve authentication from the session cookie and no longer fall back to `authToken` URL parameters.
- `dashboard/frontend/src/utils/authFetch.ts`, `AuthContext.tsx`, and the affected page-level iframe URLs now rely on same-origin cookie auth and 401 handling rather than token persistence in `localStorage`, JS-readable cookies, or transport-specific URL rewriting.
- `docs/agent/state-taxonomy-and-inventory.md` now classifies playground chat history as browser-local convenience state instead of an implicit shared product surface, so this debt no longer needs to stay open just to preserve that ambiguity.

## Validation

- `go test ./auth ./router`
  Run in `/Users/bitliu/vs/dashboard/backend`
- `npm run type-check`
  Run in `/Users/bitliu/vs/dashboard/frontend`
- `npm run lint`
  Run in `/Users/bitliu/vs/dashboard/frontend`
- `npx playwright test e2e/auth-flow.spec.ts`
  Run in `/Users/bitliu/vs/dashboard/frontend`
- `make dashboard-check`
  Run in `/Users/bitliu/vs`
- `make agent-dev ENV=cpu`
  Run in `/Users/bitliu/vs`
- `make agent-serve-local ENV=cpu`
  Run in `/Users/bitliu/vs`
- `make agent-smoke-local`
  Run in `/Users/bitliu/vs`
