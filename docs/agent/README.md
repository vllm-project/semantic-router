# Agent Harness

This directory is the human-readable entry for the repository harness.

The harness has three audiences:

- maintainers planning releases and managing GitHub work
- contributors trying to make a correct change
- coding agents resolving context and validation from changed files

## Maintainer

Start here when managing release scope, milestones, issue flow, PR review, or
architecture debt:

- [maintainer-ops.md](maintainer-ops.md)
- [plans/README.md](plans/README.md)
- [tech-debt/README.md](tech-debt/README.md)
- [tech-debt-register.md](tech-debt-register.md)
- [architecture-scorecard.md](architecture-scorecard.md)

Current maintainer rule:

- one active release plan per release
- one active debt plan for non-release debt
- generated issue/PR board under `.agent-harness/maintainer/`

## Contributor

Start here when changing code or docs:

- [repo-map.md](repo-map.md)
- [change-surfaces.md](change-surfaces.md)
- [module-boundaries.md](module-boundaries.md)
- [testing-strategy.md](testing-strategy.md)
- [feature-complete-checklist.md](feature-complete-checklist.md)
- nearest local `AGENTS.md`

Contributor-facing wrappers also live in:

- [../../CONTRIBUTING.md](../../CONTRIBUTING.md)
- [../../.github/PULL_REQUEST_TEMPLATE.md](../../.github/PULL_REQUEST_TEMPLATE.md)
- [../../.github/ISSUE_TEMPLATE/001_feature_request.yaml](../../.github/ISSUE_TEMPLATE/001_feature_request.yaml)
- [../../.github/ISSUE_TEMPLATE/002_bug_report.yaml](../../.github/ISSUE_TEMPLATE/002_bug_report.yaml)
- [../../.prowlabels.yaml](../../.prowlabels.yaml)

## Coding Agent

Start here when resolving task context mechanically:

- [context-management.md](context-management.md)
- [environments.md](environments.md)
- [local-rules.md](local-rules.md)
- [skill-catalog.md](skill-catalog.md)
- [.agents/skills/harness/SKILL.md](../../.agents/skills/harness/SKILL.md)

The default loop is:

```bash
make agent-report ENV=cpu CHANGED_FILES="..."
make agent-validate
make agent-ci-gate CHANGED_FILES="..."
```

## Governance

- [governance.md](governance.md)
- [architecture-guardrails.md](architecture-guardrails.md)
- [openai-api-contracts.md](openai-api-contracts.md)
- [reviews/README.md](reviews/README.md)
- [reviews/2026/2026-05-22-memory-assisted-review-bot.md](reviews/2026/2026-05-22-memory-assisted-review-bot.md)
- [glossary.md](glossary.md)
- [amd-local.md](amd-local.md)

## Executable Contract

- [../../tools/agent/repo-manifest.yaml](../../tools/agent/repo-manifest.yaml)
- [../../tools/agent/context-map.yaml](../../tools/agent/context-map.yaml)
- [../../tools/agent/skill-registry.yaml](../../tools/agent/skill-registry.yaml)
- [../../tools/agent/task-matrix.yaml](../../tools/agent/task-matrix.yaml)
- [../../tools/agent/e2e-profile-map.yaml](../../tools/agent/e2e-profile-map.yaml)
- [../../tools/agent/structure-rules.yaml](../../tools/agent/structure-rules.yaml)
- [../../tools/agent/maintainer-policy.yaml](../../tools/agent/maintainer-policy.yaml)
- [../../tools/make/agent.mk](../../tools/make/agent.mk)
