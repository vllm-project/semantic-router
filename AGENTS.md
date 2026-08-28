# vLLM Semantic Router Agent Entry

This file is the short entrypoint for coding agents. The detailed human-readable system of record lives in [tools/agent/docs/README.md](tools/agent/docs/README.md). The executable rule layer lives in [tools/agent/repo-manifest.yaml](tools/agent/repo-manifest.yaml), [tools/agent/test-domain-registry.yaml](tools/agent/test-domain-registry.yaml), [tools/agent/task-matrix.yaml](tools/agent/task-matrix.yaml), [tools/agent/skill-registry.yaml](tools/agent/skill-registry.yaml), [tools/agent/structure-rules.yaml](tools/agent/structure-rules.yaml), [tools/agent/maintainer-policy.yaml](tools/agent/maintainer-policy.yaml), and [tools/make/agent.mk](tools/make/agent.mk).

vLLM Semantic Router is an Envoy ExtProc request router for LLM inference. It
resolves a request-facing entrypoint to an isolated recipe, evaluates that
recipe's signals and projections, applies its decision and algorithm, then
invokes the selected backend and recipe-scoped plugins.

## Read First

1. [tools/agent/docs/README.md](tools/agent/docs/README.md)
2. [tools/agent/docs/repo-map.md](tools/agent/docs/repo-map.md)
3. [tools/agent/docs/environments.md](tools/agent/docs/environments.md)
4. [tools/agent/docs/change-surfaces.md](tools/agent/docs/change-surfaces.md)
5. `make agent-report ENV=cpu|amd CHANGED_FILES="..."`

## Task Routing

- Root startup begins with this [AGENTS.md](AGENTS.md) entrypoint.
- Full task routing, primary-skill resolution, local-rule surfacing, loop-mode guidance, and validation planning come from `make agent-report ENV=cpu|amd CHANGED_FILES="..."`.
- Follow the primary skill selected from [tools/agent/skills/](tools/agent/skills/)
  instead of treating the root entrypoint as a task-specific skill.
- `tools/agent/**` remains the canonical harness source.

If you need real AMD model deployment details instead of the minimal smoke path, also read [website/docs/installation/amd-rocm.md](website/docs/installation/amd-rocm.md) and [config/recipes/balance/config.yaml](config/recipes/balance/config.yaml).

## Repository Map

- `src/semantic-router/`: Go router, Envoy ExtProc server, routing runtime, and APIs
- `src/vllm-sr/`: Python CLI and local stack orchestration
- `config/`: canonical config, reusable fragments, runtime examples, and complete recipes
- `candle-binding/`, `ml-binding/`, `nlp-binding/`, `onnx-binding/`: inference bindings
- `dashboard/`: React frontend and Go management backend
- `deploy/`: artifacts that create or configure deployment targets
- `e2e/`: end-to-end framework and profiles
- `tools/`: build, development, release, security, smoke, and agent tooling
- `website/`: the only public documentation tree

The root contains only repository-wide contracts, community metadata, and
tool-mandated entrypoints. The executable allowlist is in
`tools/agent/structure-rules.yaml`; do not add root catch-all files.

## Supported Environments

- `cpu-local`: `make vllm-sr-dev`, then `vllm-sr serve --image-pull-policy never`
- `amd-local`: `make vllm-sr-dev VLLM_SR_PLATFORM=amd`, then `vllm-sr serve --image-pull-policy never --platform amd`
- `nvidia-local`: `VLLM_SR_PLATFORM=nvidia make vllm-sr-build`, then `vllm-sr serve --platform nvidia --config <recipe>` (selects the CUDA image + flips `use_cpu` to false, at parity with `--platform amd`; see [tools/agent/docs/nvidia-local.md](tools/agent/docs/nvidia-local.md))
- `ci-k8s`: `make e2e-test`

## Non-Negotiable Rules

- Use the local image flow for local-dev behavior. Do not invent another serve path.
- Start from one project-level primary skill. Cross-cutting guidance belongs in change surfaces, canonical docs, or maintainer support skills.
- Run the smallest relevant gate first: `make agent-validate`, `make agent-lint`, `make agent-ci-gate`, then `make agent-feature-gate`.
- Use `make agent-pr-gate` when you need a repo-native local reproduction of the baseline PR requirements.
- Drive the active task to its reported completion boundary: fix failures and rerun the applicable gates until the current change or subtask is done, and do not hand off on the first failing run.
- Treat docs-only and website-only edits as lightweight unless the task matrix says otherwise.
- Contributor workflow, issue or PR intake rules, and maintainer label taxonomy live in `CONTRIBUTING.md`, `.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/**`, and `.prowlabels.yaml`; commits intended for PRs must use `git commit -s`.
- Keep commits reviewable and logically coherent. Each commit must build and lint, describe why the change exists, and avoid unrelated drive-by cleanup.
- Keep PR blast radius aligned with the requested behavior; couple subsystems only when their contracts change together.
- Maintainer release, issue, PR, stale-work, and daily-board workflows live in [tools/agent/docs/maintainer-ops.md](tools/agent/docs/maintainer-ops.md) and write local state only under `.agent-harness/maintainer/` unless an explicit reviewed apply step mutates GitHub.
- Behavior-visible routing, startup, config, Docker, CLI, or API changes need E2E updates unless the change is a pure refactor.
- If the work needs multiple resumable loops across sessions or contributors, use the indexed current execution plans under [tools/agent/docs/plans/README.md](tools/agent/docs/plans/README.md) instead of ad hoc task notes. Historical plans are not kept in the current tree.
- If the desired architecture and the current implementation still diverge after your change, add or update the durable debt entry indexed from [tools/agent/docs/tech-debt/README.md](tools/agent/docs/tech-debt/README.md) instead of leaving the gap only in chat or PR text.
- Keep modules narrow: one main responsibility per file, small orchestrators plus helpers, interfaces only at seams.
- Legacy hotspots are debt, not precedent. Touched hotspot files must not grow in responsibility; prefer extraction-first edits.
- Read the nearest local `AGENTS.md` before editing hotspot trees under `src/semantic-router/pkg/config/`, `src/semantic-router/pkg/extproc/`, `src/vllm-sr/cli/`, `src/fleet-sim/fleet_sim/optimizer/`, `deploy/operator/api/v1alpha1/`, `deploy/operator/controllers/`, `dashboard/frontend/src/`, `dashboard/frontend/src/pages/`, `dashboard/frontend/src/components/`, and `dashboard/backend/handlers/`.

## Canonical Commands

- Harness: `make agent-bootstrap`, `make agent-validate`, `make agent-report ENV=cpu|amd CHANGED_FILES="..."`, `make agent-lint`, `make agent-ci-gate`, `make agent-pr-gate`
- Build and core: `make test-and-build-local`, `make test-semantic-router`, `make test-binding`, `make check-go-mod-tidy`
- Classifiers: `make test-category-classifier`, `make test-pii-classifier`, `make test-jailbreak-classifier`
- Runtime and E2E: `make agent-dev ENV=cpu|amd`, `make agent-serve-local ENV=cpu|amd`, `make agent-feature-gate ENV=cpu|amd CHANGED_FILES="..."`, `make agent-e2e-affected CHANGED_FILES="..."`
- Full lint: `pre-commit run --all-files`

## Rule Layers

- Entry and navigation: [tools/agent/docs/README.md](tools/agent/docs/README.md), [tools/agent/docs/governance.md](tools/agent/docs/governance.md)
- Architecture and boundaries: [tools/agent/docs/architecture-guardrails.md](tools/agent/docs/architecture-guardrails.md), nearest local `AGENTS.md`
- Testing and done criteria: [tools/agent/docs/feature-complete-checklist.md](tools/agent/docs/feature-complete-checklist.md)
- Executable contract: [tools/agent/repo-manifest.yaml](tools/agent/repo-manifest.yaml), [tools/agent/test-domain-registry.yaml](tools/agent/test-domain-registry.yaml), [tools/agent/task-matrix.yaml](tools/agent/task-matrix.yaml), [tools/agent/skill-registry.yaml](tools/agent/skill-registry.yaml), [tools/agent/structure-rules.yaml](tools/agent/structure-rules.yaml)
- Maintainer ops: [tools/agent/docs/maintainer-ops.md](tools/agent/docs/maintainer-ops.md), [tools/agent/maintainer-policy.yaml](tools/agent/maintainer-policy.yaml)

Temporary working notes can exist when needed, but they are not part of the canonical harness unless promoted into the docs or executable rule layer above.
