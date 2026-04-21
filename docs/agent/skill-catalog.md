# Skill Catalog

This document is the human-readable index for the repository's skill system.
Skills stay intentionally concise. Keep deep reference material in the linked playbooks and docs instead of copying it into each skill, and use `## Gotchas` in primary or support skills to capture repeated failure modes that need to stay visible at trigger time.

## Activation Model

- Primary skills pick the task archetype.
- Fragment skills are a reusable inventory, not an unconditional include list.
- The harness resolves active fragments from the primary-skill inventory using the impacted surfaces and active environment.
- The fallback `cross-stack-bugfix` primary resolves fragments directly from impacted surfaces instead of hard-coding the entire fragment inventory.
- Support skills are manual follow-up tools and should not expand the default context pack unless they are explicitly invoked.
- The completion checklist remains canonical repo documentation, not a default-loaded support skill.

## Primary Skills

- `harness-contract-change`
- `signal-end-to-end`
- `plugin-end-to-end`
- `config-platform-change`
- `routing-policy-change`
- `startup-chain-change`
- `dashboard-platform-change`
- `router-service-platform-change`
- `fleet-sim-change`
- `k8s-platform-change`
- `training-stack-change`
- `cross-stack-bugfix`

## Fragment Skills

- `signal-runtime`
- `routing-policy-runtime`
- `plugin-runtime`
- `router-service-platform`
- `binding-ffi`
- `python-cli-schema`
- `python-cli-runtime`
- `dashboard-platform-runtime`
- `k8s-platform-runtime`
- `fleet-sim-runtime`
- `training-stack-runtime`
- `e2e-selection`
- `local-dev-cpu`
- `local-dev-amd`

## Support Skills

- `routing-calibration-loop`
- `maintainer-issue-pr-management`
- `architecture-guardrails`
- `claude-code-install`

## Source of Truth

- Executable registry: [../../tools/agent/skill-registry.yaml](../../tools/agent/skill-registry.yaml)
- Human-readable surface map: [change-surfaces.md](change-surfaces.md)
- Supporting implementation playbooks: [playbooks/go-router.md](playbooks/go-router.md), [playbooks/rust-bindings.md](playbooks/rust-bindings.md), [playbooks/vllm-sr-cli-docker.md](playbooks/vllm-sr-cli-docker.md), [playbooks/e2e-selection.md](playbooks/e2e-selection.md)
