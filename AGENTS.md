# vLLM Semantic Router Agent Entry

This file is the short entrypoint for coding agents. The detailed human-readable system of record lives in [docs/agent/README.md](docs/agent/README.md). The executable rule layer lives in [tools/agent/repo-manifest.yaml](tools/agent/repo-manifest.yaml), [tools/agent/task-matrix.yaml](tools/agent/task-matrix.yaml), [tools/agent/skill-registry.yaml](tools/agent/skill-registry.yaml), [tools/agent/structure-rules.yaml](tools/agent/structure-rules.yaml), and [tools/make/agent.mk](tools/make/agent.mk).

## Read First

1. [docs/agent/README.md](docs/agent/README.md)
2. [docs/agent/repo-map.md](docs/agent/repo-map.md)
3. [docs/agent/environments.md](docs/agent/environments.md)
4. [docs/agent/change-surfaces.md](docs/agent/change-surfaces.md)
5. `make agent-report ENV=cpu|amd CHANGED_FILES="..."`

If you need real AMD model deployment details instead of the minimal smoke path, also read [deploy/amd/README.md](deploy/amd/README.md) and [deploy/amd/config.yaml](deploy/amd/config.yaml).

## Supported Environments

- `cpu-local`: `make vllm-sr-dev`, then `vllm-sr serve --image-pull-policy never`
- `amd-local`: `make vllm-sr-dev VLLM_SR_PLATFORM=amd`, then `vllm-sr serve --image-pull-policy never --platform amd`
- `ci-k8s`: `make e2e-test`

## Non-Negotiable Rules

- Use the local image flow for local-dev behavior. Do not invent another serve path.
- Start from a project-level primary skill. Fragment skills are support material, not the default entrypoint.
- Run the smallest relevant gate first: `make agent-validate`, `make agent-lint`, `make agent-ci-gate`, then `make agent-feature-gate`.
- Treat docs-only and website-only edits as lightweight unless the task matrix says otherwise.
- Contributor workflow and PR intake rules live in `CONTRIBUTING.md` and `.github/PULL_REQUEST_TEMPLATE.md`; commits intended for PRs must use `git commit -s`.
- Behavior-visible routing, startup, config, Docker, CLI, or API changes need E2E updates unless the change is a pure refactor.
- If the work needs multiple resumable loops across sessions or contributors, use the indexed execution plans under [docs/agent/plans/README.md](docs/agent/plans/README.md) instead of ad hoc task notes.
- If the desired architecture and the current implementation still diverge after your change, add or update the durable debt entry indexed from [docs/agent/tech-debt/README.md](docs/agent/tech-debt/README.md) instead of leaving the gap only in chat or PR text.
- Keep modules narrow: one main responsibility per file, small orchestrators plus helpers, interfaces only at seams.
- Legacy hotspots are debt, not precedent. Touched hotspot files must not grow in responsibility; prefer extraction-first edits.
- Read the nearest local `AGENTS.md` before editing hotspot trees under `src/semantic-router/pkg/config/`, `src/semantic-router/pkg/extproc/`, `src/vllm-sr/cli/`, `dashboard/frontend/src/pages/`, and `dashboard/frontend/src/components/`.

## Canonical Commands

- `make agent-bootstrap`
- `make agent-validate`
- `make agent-scorecard`
- `make agent-dev ENV=cpu|amd`
- `make agent-serve-local ENV=cpu|amd`
- `make agent-report ENV=cpu|amd CHANGED_FILES="..."`
- `make agent-lint CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-feature-gate ENV=cpu|amd CHANGED_FILES="..."`
- `make agent-e2e-affected CHANGED_FILES="..."`

## Rule Layers

- Entry and navigation: [docs/agent/README.md](docs/agent/README.md), [docs/agent/governance.md](docs/agent/governance.md)
- Architecture and boundaries: [docs/agent/architecture-guardrails.md](docs/agent/architecture-guardrails.md), nearest local `AGENTS.md`
- Testing and done criteria: [docs/agent/feature-complete-checklist.md](docs/agent/feature-complete-checklist.md)
- Executable contract: [tools/agent/repo-manifest.yaml](tools/agent/repo-manifest.yaml), [tools/agent/task-matrix.yaml](tools/agent/task-matrix.yaml), [tools/agent/skill-registry.yaml](tools/agent/skill-registry.yaml), [tools/agent/e2e-profile-map.yaml](tools/agent/e2e-profile-map.yaml), [tools/agent/structure-rules.yaml](tools/agent/structure-rules.yaml)

Temporary working notes can exist when needed, but they are not part of the canonical harness unless promoted into the docs or executable rule layer above.
