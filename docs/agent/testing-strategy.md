# Testing Strategy

This document defines the harness-side validation ladder for repository changes.

## Validation Ladder

- `make agent-validate`
  - use for harness-only changes
  - validates manifests, docs inventory, rule layering, and link portability
- `make agent-scorecard`
  - shows the current harness inventory and whether validation is passing
- `make agent-lint CHANGED_FILES="..."`
  - runs pre-commit, language lint, and structure checks for changed files
  - Go changed-file lint reuses stricter module configs when the repository defines them; `dashboard/backend` uses the same `golangci-lint` config as `make dashboard-lint`
- `make agent-ci-gate CHANGED_FILES="..."`
  - runs `agent-report`, `agent-fast-gate`, and rule-driven fast tests
- `make agent-feature-gate ENV=cpu|amd CHANGED_FILES="..."`
  - runs the CI gate, feature tests, local smoke when required, affected local E2E, and a final report

## Selection Rules

- Harness-only prose or manifest changes start with `make agent-validate`.
- Code changes start with `make agent-report` to resolve primary skill, impacted surfaces, and validation commands.
- Use the smallest gate that matches the change.
- Use `ENV=amd` when platform behavior, AMD defaults, or ROCm image selection are affected.

## Environment Expectations

- `cpu-local`
  - default local path for feature work
- `amd-local`
  - required for AMD-specific behavior and real-model deployment validation
- `ci-k8s`
  - merge-gate coverage and profile matrix validation

See [environments.md](environments.md) for the concrete commands.

## Behavior and Coverage Rules

- Behavior-visible routing, startup, config, Docker, CLI, or API changes require updated or new E2E coverage unless the change is a pure refactor.
- Documentation-only changes should not trigger local smoke or heavy E2E unless the task matrix escalates them.
- Core, common, startup-chain, Docker, or agent-execution changes may expand CI profile coverage beyond the locally affected set.
- Workflow-driven integration suites are part of the canonical validation story when they are listed in `tools/agent/e2e-profile-map.yaml`.
- The current workflow-driven suites are:
  - `vllm-sr-cli-integration` via `make vllm-sr-test-integration`
  - `memory-integration` via `make memory-test-integration`
- Manual-only Go profiles are valid durable suites, but they must be named in `manual_profile_rules` instead of existing as undocumented runner-only paths.

## Source of Truth

- Gate selection and commands: [../../tools/agent/task-matrix.yaml](../../tools/agent/task-matrix.yaml)
- Environment resolution: [../../tools/agent/repo-manifest.yaml](../../tools/agent/repo-manifest.yaml)
- E2E profile mapping: [../../tools/agent/e2e-profile-map.yaml](../../tools/agent/e2e-profile-map.yaml)
- E2E taxonomy and suite selection guidance: [playbooks/e2e-selection.md](playbooks/e2e-selection.md)
- Executable entrypoints: [../../tools/make/agent.mk](../../tools/make/agent.mk)
- Done criteria: [feature-complete-checklist.md](feature-complete-checklist.md)
