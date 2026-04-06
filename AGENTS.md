# vLLM Semantic Router Agent Entry

This file is the short entrypoint for coding agents. The detailed human-readable system of record lives in [docs/agent/README.md](docs/agent/README.md). The executable rule layer lives in [tools/agent/repo-manifest.yaml](tools/agent/repo-manifest.yaml), [tools/agent/task-matrix.yaml](tools/agent/task-matrix.yaml), [tools/agent/skill-registry.yaml](tools/agent/skill-registry.yaml), [tools/agent/structure-rules.yaml](tools/agent/structure-rules.yaml), and [tools/make/agent.mk](tools/make/agent.mk).

## Read First

1. [docs/agent/README.md](docs/agent/README.md)
2. [docs/agent/repo-map.md](docs/agent/repo-map.md)
3. [docs/agent/environments.md](docs/agent/environments.md)
4. [docs/agent/change-surfaces.md](docs/agent/change-surfaces.md)
5. `make agent-report ENV=cpu|amd CHANGED_FILES="..."`

## Native Discovery vs Routed Context

- Root startup should always discover this [AGENTS.md](AGENTS.md) entrypoint and the thin repo-native bridge at [.agents/skills/harness/SKILL.md](.agents/skills/harness/SKILL.md).
- Full task routing, primary-skill resolution, local-rule surfacing, loop-mode guidance, and validation planning still come from `make agent-report ENV=cpu|amd CHANGED_FILES="..."`.
- `tools/agent/**` remains the canonical harness source; `.agents/skills/**` is only a discovery bridge.

If you need real AMD model deployment details instead of the minimal smoke path, also read [deploy/amd/README.md](deploy/amd/README.md) and [deploy/recipes/balance.yaml](deploy/recipes/balance.yaml).

## Supported Environments

- `cpu-local`: `make vllm-sr-dev`, then `vllm-sr serve --image-pull-policy never`
- `amd-local`: `make vllm-sr-dev VLLM_SR_PLATFORM=amd`, then `vllm-sr serve --image-pull-policy never --platform amd`
- `ci-k8s`: `make e2e-test`

## Non-Negotiable Rules

- Use the local image flow for local-dev behavior. Do not invent another serve path.
- Start from a project-level primary skill. Fragment skills are support material, not the default entrypoint.
- Run the smallest relevant gate first: `make agent-validate`, `make agent-lint`, `make agent-ci-gate`, then `make agent-feature-gate`.
- Use `make agent-pr-gate` when you need a repo-native local reproduction of the baseline PR requirements.
- Drive the active task to its reported completion boundary: fix failures and rerun the applicable gates until the current change or subtask is done, and do not hand off on the first failing run.
- Treat docs-only and website-only edits as lightweight unless the task matrix says otherwise.
- Contributor workflow, issue or PR intake rules, and maintainer label taxonomy live in `CONTRIBUTING.md`, `.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/**`, and `.prowlabels.yaml`; commits intended for PRs must use `git commit -s`.
- Behavior-visible routing, startup, config, Docker, CLI, or API changes need E2E updates unless the change is a pure refactor.
- If the work needs multiple resumable loops across sessions or contributors, use the indexed execution plans under [docs/agent/plans/README.md](docs/agent/plans/README.md) instead of ad hoc task notes.
- If the desired architecture and the current implementation still diverge after your change, add or update the durable debt entry indexed from [docs/agent/tech-debt/README.md](docs/agent/tech-debt/README.md) instead of leaving the gap only in chat or PR text.
- Keep modules narrow: one main responsibility per file, small orchestrators plus helpers, interfaces only at seams.
- Legacy hotspots are debt, not precedent. Touched hotspot files must not grow in responsibility; prefer extraction-first edits.
- Read the nearest local `AGENTS.md` before editing hotspot trees indexed from [docs/agent/local-rules.md](docs/agent/local-rules.md).

## Canonical Commands

- `make agent-bootstrap`
- `make agent-validate`
- `make agent-scorecard`
- `make agent-dev ENV=cpu|amd`
- `make agent-serve-local ENV=cpu|amd`
- `make agent-report ENV=cpu|amd CHANGED_FILES="..."`
- `make agent-lint CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-pr-gate`
- `make test-and-build-local`
- `make agent-feature-gate ENV=cpu|amd CHANGED_FILES="..."`
- `make agent-e2e-affected CHANGED_FILES="..."`

## Rule Layers

- Entry and navigation: [docs/agent/README.md](docs/agent/README.md), [docs/agent/governance.md](docs/agent/governance.md)
- Architecture and boundaries: [docs/agent/architecture-guardrails.md](docs/agent/architecture-guardrails.md), nearest local `AGENTS.md`
- Testing and done criteria: [docs/agent/feature-complete-checklist.md](docs/agent/feature-complete-checklist.md)
- Executable contract: [tools/agent/repo-manifest.yaml](tools/agent/repo-manifest.yaml), [tools/agent/task-matrix.yaml](tools/agent/task-matrix.yaml), [tools/agent/skill-registry.yaml](tools/agent/skill-registry.yaml), [tools/agent/e2e-profile-map.yaml](tools/agent/e2e-profile-map.yaml), [tools/agent/structure-rules.yaml](tools/agent/structure-rules.yaml)

Temporary working notes can exist when needed, but they are not part of the canonical harness unless promoted into the docs or executable rule layer above.
