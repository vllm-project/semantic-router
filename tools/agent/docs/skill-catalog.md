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
- `project-change` (lightweight fallback; boundaries come from changed-file
  surfaces and the nearest `AGENTS.md`)

## Support Skills

Maintainer:

- `maintainer-release-ops`
- `maintainer-issue-pr-management`
- `routing-calibration-loop`

## Source of Truth

- Executable registry: [skill-registry.yaml](../skill-registry.yaml)
- Maintainer policy: [maintainer-policy.yaml](../maintainer-policy.yaml)
- Change surfaces: [change-surfaces.md](change-surfaces.md)
