# TD005 Dashboard Enterprise Console Foundations Execution Plan

This plan tracked the long-horizon remediation loop for `TD005 Dashboard Lacks Enterprise Console Foundations` and now remains as the canonical completion record for the retired workstream.

## Goal

- Evolve the dashboard from a file-backed operator UI into a secure console control plane with durable state, first-class identity, and policy-enforced operations.
- Keep the remediation recoverable across multiple implementation loops without relying on chat memory.

## Scope

- `dashboard/backend/**` control-plane APIs, storage seams, proxy behavior, and auth/session wiring
- `dashboard/frontend/**` callers that currently depend on direct file-backed config APIs or readonly-only settings
- Dashboard runtime and docs surfaces that define console behavior, validation, and rollout expectations

## Current State Snapshot

- Config history, deploy events, audit records, users, sessions, and role bindings now persist through the dedicated console store instead of living only in YAML files and backup folders.
- The primary dashboard write workflows already use the revision-backed `draft -> validate -> activate` model, while compatibility endpoints remain in place only as adapters for older callers.
- The dashboard now resolves a first-class session at `/api/auth/session`, supports local bootstrap auth plus trusted-proxy auth, and enforces `viewer/editor/operator/admin` policy at route registration time.
- Frontend config save, builder deploy, setup activation, evaluation, ML setup, and OpenClaw flows now resolve dashboard capabilities and disable or hide mutation paths when the role is insufficient.
- Embedded proxy access now enforces dashboard-origin checks, audits embedded-service entry, and mediates OpenClaw gateway tokens server-side instead of exposing raw service tokens to the browser.
- The canonical `cpu-local` smoke path now uses an explicitly API-only smoke config that clears bundled local-model defaults before merge, so `make agent-feature-gate ENV=cpu` validates dashboard startup without depending on opportunistic HF model downloads.

## Exit Criteria

- The dashboard has a coherent persistent storage model for console state and config workflows.
- Config changes flow through explicit draft, validation, activation, rollback, and audit states instead of direct file mutation as the primary control path.
- Authentication, session management, and role-based authorization exist as supported product features instead of roadmap notes.
- Embedded proxy behavior has explicit security policy, token/session mediation, and auditability appropriate for a shared enterprise console.
- Backend, frontend, docs, and validation gates are aligned well enough to retire `TD005`.

## Task List

- [x] `TD005-P01` Capture the current-state architecture, gap analysis, and staged remediation plan in a canonical execution plan.
  - Done when this file records the workstream boundaries, target state, and ordered tasks.
- [x] `TD005-P02` Define the console domain model and storage contract.
  - Done when the repo has an explicit model for console users, roles, sessions, config revisions, deployment events, secret references, and audit records.
- [x] `TD005-P03` Extract backend application seams so the console no longer hangs directly off `router.Setup` plus filesystem-coupled handlers.
  - Done when storage, config lifecycle, proxy policy, and auth/session logic have separable services or packages with narrow responsibilities.
- [x] `TD005-P04` Introduce a real console persistence layer.
  - Done when the dashboard can persist console state outside ad hoc YAML backups, with a local-friendly backend and a multi-user or cluster-ready path.
- [x] `TD005-P05` Move config workflows to a revision-based control-plane model.
  - Done when the primary workflow is draft -> validate -> activate -> rollback, and runtime YAML is treated as a generated deployment artifact rather than the console source of truth.
- [x] `TD005-P06` Add first-class authentication and session management.
  - Done when the dashboard supports a bootstrap local path for development and an OIDC-ready production path without leaving auth as a future-note.
- [x] `TD005-P07` Add role-based authorization for console actions.
  - Done when read, edit, deploy, rollback, setup activation, evaluation, and admin routes have policy-enforced access control.
- [x] `TD005-P08` Harden embedded proxy and service mediation.
  - Done when proxy behavior uses explicit upstream allowlists, CSRF or origin policy, signed or mediated service access, and auditable action boundaries instead of broad pass-through behavior.
- [x] `TD005-P09` Align frontend callers with the new control-plane APIs.
  - Done when config pages, setup wizard, readonly or settings flows, and any admin pages consume auth-aware, revision-aware APIs instead of assuming direct file mutation.
- [x] `TD005-P10` Expand validation coverage for enterprise-console behavior.
  - Done when `make dashboard-check` covers the touched backend and frontend paths, user-visible changes have E2E coverage, and the local smoke path validates the console startup model.
- [x] `TD005-P11` Complete rollout, migration, and debt retirement updates.
  - Done when docs, migration notes, and any required ADRs are updated and `TD005` can be retired from the indexed tech-debt entry set.

## Current Loop

- [x] `TD005-L01` Choose the persistence strategy and boundary.
  - Target: a `console store` abstraction with SQLite as the local default and a cluster-ready SQL backend path for shared deployments.
- [x] `TD005-L02` Define the minimal enterprise-console domain model before coding.
  - Target: `user`, `role_binding`, `session`, `config_revision`, `deploy_event`, `secret_ref`, and `audit_event` as first-class persisted concepts.
- [x] `TD005-L03` Carve out backend services without changing product behavior.
  - Target: split config lifecycle, auth/session, proxy mediation, and storage access out of `dashboard/backend/router/router.go` and the large direct handlers.
- [x] `TD005-L04` Introduce revision-backed config APIs behind a compatibility adapter.
  - Target: existing frontend flows continue to work while the backend starts persisting revisions and generating runtime YAML on activation.

## Decision Log

- Treat router-facing YAML as a runtime artifact, not the long-term source of truth for console workflows.
- Do not extend the evaluation SQLite schema into a generic console state bucket. Reuse the database engine if helpful, but keep console storage ownership and schema boundaries explicit.
- The first concrete implementation step uses a dedicated `dashboard/backend/console` package with a backend-agnostic store contract and a local SQLite implementation. Future shared-deployment backends should implement the same contract instead of widening file-backed handlers.
- Backend dependency initialization now starts from an application bootstrap object instead of wiring routes directly from raw config alone. Future config lifecycle and auth services should attach to that bootstrap seam rather than being created ad hoc inside `router.Setup`.
- The next L03 seam is no longer blocked on `router/router.go` or `config/config.go` file size. Route registration and config flag loading now live behind smaller helpers, so the remaining extraction work should focus on moving file-backed config lifecycle behavior out of the direct handlers instead of adding more logic to those hotspots.
- File-backed config, deploy, rollback, and setup activation now run through `dashboard/backend/configlifecycle`, with the large handlers reduced to compatibility adapters. Successful compatibility mutations also persist `config_revision`, `deploy_event`, and `audit_event` records into the console store so the revision model can grow behind stable APIs before the frontend switches over.
- Compatibility reads for current config and setup state now prefer the latest active persisted revision when one exists, with filesystem fallback preserved for bootstrap and non-console paths. The remaining L04 gap is exposing revision-native list/read/activate APIs instead of only upgrading the old file-shaped endpoints behind the scenes.
- The dashboard backend now exposes revision-native read endpoints for config history and the current active revision, while legacy `/api/router/config/all`, `/yaml`, and setup-state reads continue to serve compatibility clients. Activation is still compatibility-driven, so L04 remains open until revision-native write and promote flows replace the file-first handlers.
- The dashboard backend now also exposes a revision-native activation endpoint that promotes a persisted revision into the runtime config, records deploy and audit events, and updates active-revision ordering by activation time. L04 still remains open because draft creation, validation, and revision-first write flows are not the default source of truth yet.
- The dashboard backend now exposes revision-native draft-save and validate endpoints that persist `draft -> validated` state inside the console store without mutating runtime files. Builder deploy, config-page saves, and setup activation now call revision-native write APIs directly, while the legacy compatibility endpoints remain as adapters for older callers.
- Compatibility `config/update`, `config/deploy`, and `config/rollback` writes now run through the revision save -> validate -> activate state machine whenever the console store is configured, and the primary dashboard write workflows no longer depend on those compatibility endpoints as their source of truth. `TD005-L04` is considered complete, while `TD005-P09` remains open for the rest of the frontend surface and auth-aware route alignment.
- Dashboard auth now runs through a dedicated `dashboard/backend/auth` service with two supported modes: `bootstrap` for local development and `proxy` for production deployments behind a trusted auth gateway. The frontend resolves `/api/auth/session` at startup, and the backend persists users plus sessions in the console store instead of treating auth as a future note.
- Console authorization now uses the normalized `viewer/editor/operator/admin` role set at route registration time. Config, setup, evaluation, ML pipeline, MCP, OpenClaw, and embedded proxy routes all terminate identity and policy at the dashboard backend instead of relying only on readonly mode.
- Embedded proxy access now requires an authenticated dashboard session, and router `Authorization` pass-through is no longer on by default. Shared deployments can still opt in explicitly with `DASHBOARD_PROXY_FORWARD_AUTH=true` when upstream token forwarding is required.
- Embedded service mediation now also enforces same-origin browser access for `/api/*` and `/embedded/*`, records embedded-service access audits, and keeps OpenClaw gateway tokens on the server side instead of handing them to the browser.
- The remaining frontend control surfaces now resolve `/api/auth/session` and reflect capability boundaries directly in the UI, including evaluation actions, ML pipeline execution, OpenClaw admin access, and navigation visibility for admin-only surfaces.
- The default local smoke configs now explicitly blank the bundled prompt-guard, classifier, feedback, hallucination, and embedding model references inherited from `router-defaults.yaml`, keeping feature-gate startup focused on runtime health instead of external model downloads.
- `make agent-feature-gate ENV=cpu` now completes successfully for the TD005 surface, including `make agent-dev`, `make agent-serve-local`, and `make agent-smoke-local` with the minimal smoke config.
- Favor a strangler migration over a rewrite. Existing file-backed handlers can survive temporarily as adapters while new control-plane services take over the state machine.
- Terminate identity and authorization at the dashboard backend. Embedded services should receive mediated access, not unconstrained pass-through user credentials as the default model.
- Preserve the repo's local-dev ergonomics. Enterprise-console foundations must still support a simple local bootstrap path rather than forcing production-only infrastructure for basic development.
- Treat secret material as references or write-only updates in console APIs. Do not make full secret round-tripping part of the normal config read path.

## Follow-up Debt / ADR Links

- [../tech-debt/TD005-dashboard-enterprise-console-foundations.md](../tech-debt/TD005-dashboard-enterprise-console-foundations.md)
- [../tech-debt-register.md](../tech-debt-register.md)
- Related debt intersections: `TD001 Config Surface Fragmentation Across Router, CLI, K8s, and Dashboard`, `TD004 Local-First CLI and Kubernetes Lifecycle Still Feel Split`
- [../adr/README.md](../adr/README.md)
