# Skill Catalog

This document is the human-readable index for the repository's skill system.
Skills stay intentionally concise. Keep deep reference material in the linked playbooks and docs instead of copying it into each skill, and use `## Gotchas` in primary or support skills to capture repeated failure modes that need to stay visible at trigger time.

## Primary Skills

- `harness-contract-change`
- `signal-end-to-end`
- `plugin-end-to-end`
- `header-contract-change`
- `config-platform-change`
- `decision-logic-change`
- `algorithm-selection-change`
- `startup-chain-change`
- `dashboard-surface-change`
- `dashboard-console-platform-change`
- `router-service-platform-change`
- `fleet-sim-change`
- `k8s-operator-change`
- `deployment-profile-change`
- `training-stack-change`
- `cross-stack-bugfix`

## Fragment Skills

- `harness-governance`
- `signal-runtime`
- `decision-logic`
- `algorithm-selection`
- `plugin-runtime`
- `router-service-platform`
- `binding-ffi`
- `python-cli-schema`
- `python-cli-runtime`
- `dashboard-config-ui`
- `dashboard-console-backend`
- `topology-visualization`
- `playground-reveal`
- `dsl-crd`
- `k8s-operator`
- `deployment-profile-stack`
- `fleet-sim-runtime`
- `training-stack-runtime`
- `e2e-selection`
- `local-dev-cpu`
- `local-dev-amd`
- `architecture-guardrails`

## Support Skills

- `routing-calibration-loop`
- `feature-complete-checklist`
- `maintainer-issue-pr-management`

## Source of Truth

- Executable registry: [../../tools/agent/skill-registry.yaml](../../tools/agent/skill-registry.yaml)
- Human-readable surface map: [change-surfaces.md](change-surfaces.md)
- Supporting implementation playbooks: [playbooks/go-router.md](playbooks/go-router.md), [playbooks/rust-bindings.md](playbooks/rust-bindings.md), [playbooks/vllm-sr-cli-docker.md](playbooks/vllm-sr-cli-docker.md), [playbooks/e2e-selection.md](playbooks/e2e-selection.md)
