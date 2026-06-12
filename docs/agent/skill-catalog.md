# Skill Catalog

Skills are routed instructions for coding agents and maintainer workflows.

The physical source remains `tools/agent/skills/**`; `.agents/skills/**` is only
the native discovery bridge.

## Audience Model

- `coding-agent`: default changed-file routing and validation.
- `maintainer`: release, milestone, issue, PR, and board workflows.
- `contributor`: human-facing workflow support.

## Primary Skills

Default coding-agent routing:

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

## Support Skills

Maintainer:

- `maintainer-release-ops`
- `maintainer-issue-pr-management`
- `routing-calibration-loop`
- `review-brief-authoring`

## Source of Truth

- Executable registry: [../../tools/agent/skill-registry.yaml](../../tools/agent/skill-registry.yaml)
- Maintainer policy: [../../tools/agent/maintainer-policy.yaml](../../tools/agent/maintainer-policy.yaml)
- Change surfaces: [change-surfaces.md](change-surfaces.md)
