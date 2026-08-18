# Skill Catalog

Skills are routed instructions for coding agents and maintainer workflows. All
skill content lives under `tools/agent/skills/**`. Repository-wide bootstrap
instructions live in the root `AGENTS.md`, not in a second discovery skill.

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

## Source of Truth

- Executable registry: [../../tools/agent/skill-registry.yaml](../../../tools/agent/skill-registry.yaml)
- Maintainer policy: [../../tools/agent/maintainer-policy.yaml](../../../tools/agent/maintainer-policy.yaml)
- Change surfaces: [change-surfaces.md](change-surfaces.md)
