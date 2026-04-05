# Agent Docs Index

This directory is the human-readable system of record for the repository's agent harness.

`AGENTS.md` is the short entrypoint. Detailed rules, architecture guidance, and validation policy live here.

## Start Here

- [context-management.md](context-management.md)
  - task-first progressive disclosure, loop-mode guidance, and `agent-report` context-pack flow
- [repo-map.md](repo-map.md)
  - repo layout, hotspots, entrypoints, and high-risk areas
- [environments.md](environments.md)
  - supported local and CI environments
- [amd-local.md](amd-local.md)
  - AMD-specific local workflow details and real-model deployment references
- [change-surfaces.md](change-surfaces.md)
  - project-level change taxonomy used by skills and reports
- [feature-complete-checklist.md](feature-complete-checklist.md)
  - done criteria, validation expectations, and reporting format
- [testing-strategy.md](testing-strategy.md)
  - validation ladder, gate selection, and coverage expectations

## Governance and Structure

- [governance.md](governance.md)
  - rule layering, source-of-truth policy, and working-note policy
- [plans/README.md](plans/README.md)
  - when to use execution plans for long-horizon completion loops, how they differ from ADRs/debt, and the current plan inventory
- [tech-debt-register.md](tech-debt-register.md)
  - landing page and policy for repository technical debt
- [tech-debt/README.md](tech-debt/README.md)
  - per-item debt inventory, template, and current entry set
- [state-taxonomy-and-inventory.md](state-taxonomy-and-inventory.md)
  - canonical inventory of runtime and dashboard state surfaces, durability levels, and recovery expectations
- [glossary.md](glossary.md)
  - shared terminology for skills, surfaces, and harness layers
- [adr/README.md](adr/README.md)
  - when to use ADRs, how they differ from execution plans and debt, and the current ADR inventory
- [architecture-guardrails.md](architecture-guardrails.md)
  - file shape, module boundaries, and refactor ratchets
- [module-boundaries.md](module-boundaries.md)
  - subsystem seams, hotspot boundary rules, and where to look for local supplements
- [local-rules.md](local-rules.md)
  - indexed local `AGENTS.md` files for hotspot-specific guidance

## Task-Specific Guidance

- [skill-catalog.md](skill-catalog.md)
  - human-readable index of primary, fragment, and support skills
- [playbooks/go-router.md](playbooks/go-router.md)
- [playbooks/rust-bindings.md](playbooks/rust-bindings.md)
- [playbooks/vllm-sr-cli-docker.md](playbooks/vllm-sr-cli-docker.md)
- [playbooks/e2e-selection.md](playbooks/e2e-selection.md)
- nearest local `AGENTS.md` files under hotspot directories

## Executable Contract

- [../../tools/agent/repo-manifest.yaml](../../tools/agent/repo-manifest.yaml)
- [../../tools/agent/context-map.yaml](../../tools/agent/context-map.yaml)
- [../../tools/agent/skill-registry.yaml](../../tools/agent/skill-registry.yaml)
- [../../tools/agent/task-matrix.yaml](../../tools/agent/task-matrix.yaml)
- [../../tools/agent/e2e-profile-map.yaml](../../tools/agent/e2e-profile-map.yaml)
- [../../tools/agent/structure-rules.yaml](../../tools/agent/structure-rules.yaml)
- [../../tools/make/agent.mk](../../tools/make/agent.mk)
- Runtime entrypoints:
  - `make agent-validate`
  - `make agent-scorecard`
  - `make agent-report ENV=cpu CHANGED_FILES="..."`
    - emits validation commands, loop policy, and a compact task-first context pack by default
  - `make agent-pr-gate`
    - reproduces the baseline PR requirements locally
  - `make test-and-build-local`
    - reproduces the CI `Test And Build` job locally

## Contributor Interface

- [../../AGENTS.md](../../AGENTS.md)
- [../../CONTRIBUTING.md](../../CONTRIBUTING.md)
  - contributor workflow, validation expectations, and the `git commit -s` signoff requirement
- [../../.github/copilot-instructions.md](../../.github/copilot-instructions.md)
- [../../.github/PULL_REQUEST_TEMPLATE.md](../../.github/PULL_REQUEST_TEMPLATE.md)
  - PR title classification, affected-module context, required validation fields, and the review checklist
- [../../.github/ISSUE_TEMPLATE/001_feature_request.yaml](../../.github/ISSUE_TEMPLATE/001_feature_request.yaml)
- [../../.github/ISSUE_TEMPLATE/002_bug_report.yaml](../../.github/ISSUE_TEMPLATE/002_bug_report.yaml)
- [../../.prowlabels.yaml](../../.prowlabels.yaml)
  - issue and triage label taxonomy used for maintainer-managed issue intake
