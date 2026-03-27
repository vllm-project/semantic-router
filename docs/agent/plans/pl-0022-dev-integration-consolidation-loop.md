# Dev Integration Consolidation Loop

## Goal

- Converge the repository on one devops-facing integration model: `make vllm-sr-dev`, then `vllm-sr` owns docker or kind-backed kubernetes environment bring-up, and both environments run the same shared `dashboard` and `core routing` regression suites.
- Remove non-core environment fanout from the steady-state PR path so the default integration contract is the supported `docker | k8s(kind)` regression surface, not a broad profile matrix.
- Keep the loop resumable from repository state alone by tying execution progress to [TD037](../tech-debt/td-037-dev-integration-env-ownership-and-shared-suite-topology.md).

## Scope

- `docs/agent/plans/**`
- `docs/agent/tech-debt/**`
- `tools/agent/repo-manifest.yaml`
- `tools/agent/e2e-profile-map.yaml`
- `.github/workflows/ci-changes.yml`
- `.github/workflows/integration-test-*.yml`
- `tools/make/{docker,e2e}.mk`
- `src/vllm-sr/cli/**`
- `e2e/pkg/framework/**`
- `e2e/pkg/cluster/**`
- `e2e/pkg/fixtures/**`
- `e2e/pkg/testcases/**`
- `e2e/pkg/testmatrix/**`
- `e2e/profiles/**`
- `e2e/testing/vllm-sr-cli/**`

## Exit Criteria

- The canonical harness describes only two steady-state dev regression environments: `docker` and kind-backed `k8s`, both launched through `vllm-sr`.
- `vllm-sr serve --target k8s` owns Kind lifecycle, local image selection or load, and the `kubernetes` baseline deployment defaults needed for local regression.
- `dashboard` and `core routing` are explicit shared suites that can run against both supported environments through one environment-agnostic transport or session abstraction.
- Non-core environment profiles such as `aibrix`, `llm-d`, `istio`, and similar stack-specific variants are removed from the steady-state PR path and either retired or isolated behind clearly documented manual or nightly exceptions.
- The separate CLI integration workflow no longer duplicates the default docker regression contract; any remaining feature-specific integration workflows are documented as intentional exceptions rather than hidden parallel defaults.
- [TD037](../tech-debt/td-037-dev-integration-env-ownership-and-shared-suite-topology.md) is retired or materially narrowed because the environment ownership and shared-suite topology actually converged.

## Task List

- [x] `DIC001` Create the indexed execution plan and refresh `TD037` with the post-trim remaining-gap inventory.
  - Done when the debt record and plan both describe the same remaining work after the default `kubernetes + dashboard` baseline reduction.
- [ ] `DIC002` Lock the steady-state integration topology in canonical docs and harness manifests.
  - Done when the repo explicitly describes `docker` and `k8s(kind)` as the only default dev regression environments, and classifies every other current workflow or profile as remove, retain-as-exception, or out-of-scope feature coverage.
- [ ] `DIC003` Define the `vllm-sr`-owned kind environment contract.
  - Done when the intended `vllm-sr` responsibilities for cluster create or delete, kubeconfig handling, local image load, namespace defaults, and teardown semantics are documented before code moves.
- [ ] `DIC004` Move kind and local-image lifecycle ownership out of the Go E2E runner and into `vllm-sr` or a shared runtime support seam.
  - Done when the repository no longer relies on `e2e/pkg/framework/runner.go` as the primary owner of kind-backed environment bootstrap for the default dev regression path.
- [ ] `DIC005` Introduce an environment-agnostic shared-suite transport seam.
  - Done when `dashboard` and `core routing` test execution no longer requires Kubernetes-only `Clientset`, `RestConfig`, or port-forward helpers as the default transport contract.
- [ ] `DIC006` Rebind the shared suites to both supported environments and collapse duplicate docker-side regression paths.
  - Done when docker-backed regression coverage flows through the same named suite contract as kubernetes-backed regression, and the old workflow-only CLI integration path is either absorbed or explicitly justified as a narrow exception.
- [ ] `DIC007` Remove or demote non-core environment profiles from the steady-state PR path.
  - Done when `aibrix`, `llm-d`, `istio`, `production-stack`, `dynamic-config`, `routing-strategies`, `multi-endpoint`, `authz-rbac`, `streaming`, and `ml-model-selection` are no longer part of the default PR integration topology and the manifests or workflow selectors agree on their new status.
- [ ] `DIC008` Run the final validation ladder, update the residual debt state, and close or narrow the loop.
  - Done when the changed-file validation path passes, the remaining divergence is either retired or re-scoped in `TD037`, and the plan records the close-out state.

## Current Loop

- Loop status: opened on 2026-03-27.
- Completed in this loop:
  - audited the current `vllm-sr` runtime, Go E2E runner, workflow inventory, and existing debt or plan records against the intended `docker | k8s(kind)` regression model
  - refreshed [TD037](../tech-debt/td-037-dev-integration-env-ownership-and-shared-suite-topology.md) so it reflects the work that remains after the `kubernetes + dashboard` PR-baseline trim and the `ai-gateway -> kubernetes` profile-key rename
  - created and indexed this execution plan as the durable loop tracker for the remaining consolidation work
- Next loop focus:
  - execute `DIC002` first by classifying every remaining integration workflow or profile into default regression, legacy exception, or removal candidate before any implementation refactor starts
- Commands run:
  - startup doc reads for `AGENTS.md`, `docs/agent/{README.md,governance.md,plans/README.md,tech-debt/README.md}`, `.agents/skills/harness/SKILL.md`, and `tools/agent/skills/harness-contract-change/SKILL.md`
  - `make agent-report ENV=cpu CHANGED_FILES="docs/agent/tech-debt/td-037-dev-integration-env-ownership-and-shared-suite-topology.md docs/agent/tech-debt/README.md docs/agent/plans/pl-0002-e2e-rationalization-roadmap.md docs/agent/plans/pl-0022-dev-integration-consolidation-loop.md tools/agent/repo-manifest.yaml tools/agent/e2e-profile-map.yaml .github/workflows/integration-test-k8s.yml src/vllm-sr/cli/commands/runtime.py src/vllm-sr/cli/k8s_backend.py e2e/pkg/framework/runner.go e2e/pkg/fixtures/session.go e2e/pkg/testmatrix/testcases.go"`
  - one broad `codebase-retrieval` query for the integration architecture, runtime ownership split, workflow inventory, shared-suite coupling, and existing debt or plan coverage
- Key findings:
  - the default Kubernetes PR baseline is already reduced to `kubernetes` and `dashboard`, but the targeted matrix still retains multiple non-core environment profiles
  - `vllm-sr` still owns docker lifecycle only; Kind cluster creation and local image loading remain in the Go E2E runner
  - the named shared suites already exist in `e2e/pkg/testmatrix/testcases.go`, but their execution path still assumes Kubernetes port-forward and kube-client fixtures
  - the repository still has separate workflow-driven docker and memory integration paths, so the desired one-entrypoint regression model does not exist yet

## Decision Log

- 2026-03-27: This loop is a successor to the completed generic E2E rationalization work in [PL-0002](pl-0002-e2e-rationalization-roadmap.md); it focuses on the narrower remaining gap tracked by `TD037` instead of reopening the already-completed harness rationalization workstream.
- 2026-03-27: Keep the external baseline profile key as `kubernetes`, while allowing the underlying stack implementation to continue using the `ai-gateway` subtree until environment ownership and shared-suite convergence are complete.
- 2026-03-27: Treat non-core environment profiles as removal or exception candidates, not as part of the steady-state dev regression contract.
- 2026-03-27: Do not remove feature-specific workflows blindly; first classify whether they belong to the default regression model or to a separate, explicitly documented feature-validation surface.
- 2026-03-27: Environment ownership should move into `vllm-sr` before the repository claims that docker and kubernetes share one dev regression story.

## Follow-up Debt / ADR Links

- [TD037](../tech-debt/td-037-dev-integration-env-ownership-and-shared-suite-topology.md)
- [TD004](../tech-debt/td-004-python-cli-kubernetes-workflow-separation.md)
- [PL-0002](pl-0002-e2e-rationalization-roadmap.md)
- [PL-0020](pl-0020-local-runtime-topology-separation.md)
- [PL-0021](pl-0021-local-runtime-three-image-rollout.md)
