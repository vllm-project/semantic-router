# TD005 Dashboard Enterprise Console Foundations Execution Plan

This plan tracks the long-horizon remediation loop for `TD005 Dashboard Lacks Enterprise Console Foundations`.

## Goal

- Evolve the dashboard from a file-backed operator UI into a secure console control plane with durable state, first-class identity, and policy-enforced operations.
- Keep the remediation recoverable across multiple implementation loops without relying on chat memory.

## Scope

- `dashboard/backend/**` control-plane APIs, storage seams, proxy behavior, and auth/session wiring
- `dashboard/frontend/**` callers that currently depend on direct file-backed config APIs or readonly-only settings
- Dashboard runtime and docs surfaces that define console behavior, validation, and rollout expectations

## Current State Snapshot

- Config read, update, deploy, rollback, and setup activation all operate by reading or mutating local YAML files such as `config.yaml` and `.vllm-sr/router-defaults.yaml`.
- Deploy history is a local backup folder under `.vllm-sr/config-backups`, which is useful for recovery but is not a real persistent control-plane model.
- The dashboard has one isolated SQLite store for evaluation tasks, but config workflows, identity, sessions, and audit data are not persisted in a unified console store.
- Frontend callers treat `/api/router/config/*`, `/api/setup/*`, and `/api/settings` as the control plane, while the backend handlers remain tightly coupled to filesystem state and `router.Setup`.
- Proxying normalizes CORS/CSP and can forward `Authorization`, but the dashboard has no login, no session boundary, no RBAC enforcement, and no signed mediation for embedded services.

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
- [ ] `TD005-P03` Extract backend application seams so the console no longer hangs directly off `router.Setup` plus filesystem-coupled handlers.
  - Done when storage, config lifecycle, proxy policy, and auth/session logic have separable services or packages with narrow responsibilities.
- [ ] `TD005-P04` Introduce a real console persistence layer.
  - Done when the dashboard can persist console state outside ad hoc YAML backups, with a local-friendly backend and a multi-user or cluster-ready path.
- [ ] `TD005-P05` Move config workflows to a revision-based control-plane model.
  - Done when the primary workflow is draft -> validate -> activate -> rollback, and runtime YAML is treated as a generated deployment artifact rather than the console source of truth.
- [ ] `TD005-P06` Add first-class authentication and session management.
  - Done when the dashboard supports a bootstrap local path for development and an OIDC-ready production path without leaving auth as a future-note.
- [ ] `TD005-P07` Add role-based authorization for console actions.
  - Done when read, edit, deploy, rollback, setup activation, evaluation, and admin routes have policy-enforced access control.
- [ ] `TD005-P08` Harden embedded proxy and service mediation.
  - Done when proxy behavior uses explicit upstream allowlists, CSRF or origin policy, signed or mediated service access, and auditable action boundaries instead of broad pass-through behavior.
- [ ] `TD005-P09` Align frontend callers with the new control-plane APIs.
  - Done when config pages, setup wizard, readonly or settings flows, and any admin pages consume auth-aware, revision-aware APIs instead of assuming direct file mutation.
- [ ] `TD005-P10` Expand validation coverage for enterprise-console behavior.
  - Done when `make dashboard-check` covers the touched backend and frontend paths, user-visible changes have E2E coverage, and the local smoke path validates the console startup model.
- [ ] `TD005-P11` Complete rollout, migration, and debt retirement updates.
  - Done when docs, migration notes, and any required ADRs are updated and `TD005` can be retired from the indexed tech-debt entry set.

## Current Loop

- [x] `TD005-L01` Choose the persistence strategy and boundary.
  - Target: a `console store` abstraction with SQLite as the local default and a cluster-ready SQL backend path for shared deployments.
- [x] `TD005-L02` Define the minimal enterprise-console domain model before coding.
  - Target: `user`, `role_binding`, `session`, `config_revision`, `deploy_event`, `secret_ref`, and `audit_event` as first-class persisted concepts.
- [ ] `TD005-L03` Carve out backend services without changing product behavior.
  - Target: split config lifecycle, auth/session, proxy mediation, and storage access out of `dashboard/backend/router/router.go` and the large direct handlers.
- [ ] `TD005-L04` Introduce revision-backed config APIs behind a compatibility adapter.
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
- The dashboard backend now exposes revision-native draft-save and validate endpoints that persist `draft -> validated` state inside the console store without mutating runtime files. L04 still remains open because the legacy compatibility update/deploy endpoints are still the default write path and the frontend has not switched to revision-first workflows.
- Favor a strangler migration over a rewrite. Existing file-backed handlers can survive temporarily as adapters while new control-plane services take over the state machine.
- Terminate identity and authorization at the dashboard backend. Embedded services should receive mediated access, not unconstrained pass-through user credentials as the default model.
- Preserve the repo's local-dev ergonomics. Enterprise-console foundations must still support a simple local bootstrap path rather than forcing production-only infrastructure for basic development.
- Treat secret material as references or write-only updates in console APIs. Do not make full secret round-tripping part of the normal config read path.

## Follow-up Debt / ADR Links

- [../tech-debt/TD005-dashboard-enterprise-console-foundations.md](../tech-debt/TD005-dashboard-enterprise-console-foundations.md)
- [../tech-debt-register.md](../tech-debt-register.md)
- Related debt intersections: `TD001 Config Surface Fragmentation Across Router, CLI, K8s, and Dashboard`, `TD004 Local-First CLI and Kubernetes Lifecycle Still Feel Split`
- [../adr/README.md](../adr/README.md)
