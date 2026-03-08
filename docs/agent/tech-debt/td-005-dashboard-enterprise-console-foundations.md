# TD005: Dashboard Lacks Enterprise Console Foundations

## Status

Closed

## Scope

dashboard product architecture

## Summary

The dashboard now has a persistent console store, revision-based config workflows, bootstrap or proxy-backed auth sessions, route-level RBAC, same-origin embedded proxy mediation, capability-aware frontend control surfaces, and a canonical local smoke path that validates the minimal console startup model without opportunistic Hub downloads. The original enterprise-console foundation gap has been materially retired.

## Evidence

- [dashboard/README.md](../../../dashboard/README.md)
- [docs/agent/environments.md](../environments.md)
- [docs/agent/playbooks/vllm-sr-cli-docker.md](../playbooks/vllm-sr-cli-docker.md)
- [dashboard/backend/config/config.go](../../../dashboard/backend/config/config.go)
- [dashboard/backend/evaluation/db.go](../../../dashboard/backend/evaluation/db.go)
- [dashboard/backend/router/router.go](../../../dashboard/backend/router/router.go)

## Why It Matters

- The dashboard no longer depends on roadmap-only auth, policy, or control-plane foundations to operate as a shared console surface.
- Future dashboard gaps should now be tracked as narrower follow-on debt items instead of keeping TD005 open as a catch-all architecture placeholder.

## Desired End State

- Dashboard state and config persistence live behind a revision-based console control-plane model.
- Authentication, authorization, and user/session management exist as supported product capabilities.
- Canonical local smoke validates dashboard startup without requiring opportunistic external model downloads.

## Exit Criteria

- The dashboard has a coherent persistent storage model for console state and config workflows.
- Auth, login/session, and user/role controls exist as supported product features rather than roadmap notes.
- Canonical rollout docs and `cpu-local` smoke validation align with the shipped enterprise-console behavior.
