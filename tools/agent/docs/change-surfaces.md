# Change Surfaces

This document defines the project-level surfaces used by skills, reports, and validation.

## `router_config_contract`

- Router-side config schema and shared config files consumed directly by the runtime.
- The canonical Go types and routing registries generate one tracked artifact,
  `src/semantic-router/pkg/configschema/router-config-v0.3.schema.json`.
  CLI and container builds stage that artifact without tracked mirrors; the
  Dashboard frontend imports it directly. Do not maintain a parallel field or
  discriminator inventory.
- Typical paths: `src/semantic-router/pkg/config/**`,
  `src/semantic-router/pkg/configschema/**`, `config/**/*.yaml`
- Task rules: `router-core`, `repo-docs`

## `signal_runtime`

- Signal extraction from request or response content, including heuristic, semantic, or learned text-understanding paths.
- Typical paths: `src/semantic-router/pkg/classification/**`, `req_filter_classification.go`
- Task rules: `router-core`

## `decision_logic`

- Boolean control logic that combines signals and route conditions into decision matches.
- Typical paths: `src/semantic-router/pkg/decision/**`, `req_filter_decision*.go`
- Task rules: `router-core`

## `routing_policy`

- Router-side policy after signal extraction, covering matched-decision logic,
  candidate-model selection, and multi-model execution.
- Typical paths: `src/semantic-router/pkg/decision/**`,
  `src/semantic-router/pkg/modelselection/**`, `src/semantic-router/pkg/looper/**`,
  `src/semantic-router/pkg/projectiontrace/**`
- Task rules: `router-core`

## `algorithm_selection`

- Per-decision candidate-model selection after a decision matches.
- Typical paths: `src/semantic-router/pkg/modelselection/**`,
  `src/semantic-router/pkg/selection/**`, `src/semantic-router/pkg/looper/**`
- Task rules: `router-core`

## `plugin_runtime`

- Post-decision processing such as cache behavior, prompt rewriting, and request or response handling owned by the extproc plugin chain.
- Typical paths: `src/semantic-router/pkg/extproc/req_filter_*.go`, `processor_req_body_*.go`, `processor_res_body_*.go`
- Task rules: `router-core`

## `router_service_platform`

- Router-side service, API, storage, authz, context, telemetry, provider, and
  runtime support modules outside the config, decision, selection, and extproc
  plugin chains.
- Typical paths: `src/semantic-router/pkg/apiserver/**`, `authz/**`,
  `contextcompression/**`, `memory/**`, `modelruntime/**`, `routerruntime/**`,
  `sessiontelemetry/**`, `responseapi/**`, `publicmodels/**`
- Task rules: `router-core`

## `native_binding`

- Rust/cgo/onnx/native model bindings used by runtime signals, classifiers, or training artifacts.
- Typical paths: `candle-binding/**`, `ml-binding/**`, `nlp-binding/**`, `onnx-binding/**`
- Task rules: `rust-bindings`, `router-core`

## `response_headers`

- `x-vsr-*` header constants, router emission, dashboard reveal/display allowlists, and user-visible header contracts.
- Typical paths: `src/semantic-router/pkg/headers/**`, `processor_res_header.go`, `HeaderDisplay.tsx`, `ChatComponent*.tsx`
- Task rules: `router-core`, `dashboard`

## `python_cli_schema`

- Python CLI typed schema, parser, validator, migration, and config translation contracts.
- Typical paths: `src/vllm-sr/cli/models.py`, `parser.py`, `validator.py`, `config_migration.py`
- Task rules: `vllm-sr-cli`

## `python_cli_runtime`

- Python CLI command orchestration, local image management, serve or status flows, and startup wiring.
- Typical paths: `src/vllm-sr/cli/main.py`, `core.py`, `docker_cli.py`, `commands/**`
- Task rules: `vllm-sr-cli`

## `dashboard_platform`

- Dashboard frontend and backend surfaces that present, configure, or manage router behavior through the console UI.
- Typical paths: `dashboard/frontend/**`, `dashboard/backend/**`, `dashboard/README.md`
- Task rules: `dashboard`

## `dashboard_config_ui`

- Dashboard config editing, schema-driven forms, builder flows, and config-oriented frontend state.
- Typical paths: `dashboard/frontend/src/pages/ConfigPage*.tsx`, `BuilderPage*.tsx`, `DslEditorPage*.tsx`, `SetupWizard*.tsx`
- Task rules: `dashboard`

## `dashboard_console_backend`

- Dashboard backend, control-plane APIs, persistence, auth/session wiring, and console-side server behavior.
- Typical paths: `dashboard/backend/**`, `dashboard/README.md`
- Task rules: `dashboard`

## `topology_visualization`

- Topology parsing, graph layout, topology APIs, and highlighted decision-path visualization.
- Typical paths: `dashboard/frontend/src/pages/topology/**`, `dashboard/backend/handlers/topology.go`
- Task rules: `dashboard`

## `playground_reveal`

- Playground chat rendering, reveal overlays, and user-visible route metadata presentation.
- Typical paths: `PlaygroundPage.tsx`, `ChatComponent*.tsx`, `HeaderDisplay.tsx`, `ThinkingAnimation.tsx`
- Task rules: `dashboard`

## `dsl_crd`

- DSL compiler/decompiler and translation layers that bridge router config to Kubernetes-facing forms.
- Typical paths: `src/semantic-router/pkg/dsl/**`, `src/semantic-router/pkg/k8s/**`
- Task rules: `router-core`, `operator-stack`, `e2e-framework`

## `k8s_operator`

- Operator APIs, CRDs, controller-facing config translation, and Kubernetes deployment control-plane behavior.
- Typical paths: `deploy/operator/**`, `deploy/kubernetes/crds/**`, `src/semantic-router/pkg/apis/**`
- Task rules: `operator-stack`, `e2e-framework`

## `deployment_profile_stack`

- Kubernetes deployment profiles, stack manifests, and profile-owned platform resources outside operator CRDs.
- Typical paths: `deploy/helm/**`, `deploy/kubernetes/response-api/**`, `deploy/kubernetes/ai-gateway/**`, `deploy/kubernetes/observability/**`, `deploy/kubernetes/streaming/**`
- Task rules: `helm-chart`, `e2e-framework`

## `k8s_platform`

- Kubernetes-facing operator, CRD, deployment-profile, and DSL translation surfaces for semantic-router platform integration.
- Typical paths: `deploy/helm/**`, `deploy/operator/**`, `deploy/kubernetes/**`, `src/semantic-router/pkg/apis/**`, `src/semantic-router/pkg/dsl/**`, `src/semantic-router/pkg/k8s/**`
- Task rules: `helm-chart`, `operator-stack`, `e2e-framework`

## `fleet_sim_runtime`

- Fleet simulator package, API service, release workflow, and simulator-owned docs or assets that must stay runnable as one subsystem.
- Typical paths: `src/fleet-sim/**`, `website/docs/fleet-sim/**`, `.github/workflows/pypi-publish-vllm-sr-sim.yml`
- Task rules: `fleet-sim`, `repo-docs`

## `training_stack`

- Training-stack workflows, selector or embedding artifacts, evaluation scripts, and runtime-facing training outputs under `src/training`.
- Typical paths: `src/training/**`, `tools/make/models.mk`, `tools/models/train-mmbert32k-gpu.sh`, `website/docs/training/**`
- Task rules: `training-stack`, `repo-docs`

## `docs_examples`

- User-facing docs, examples, presets, and reference configs.
- Typical paths: `website/**`, `config/**`
- Task rules: `repo-docs`, `training-stack`

## Development harness

- Ownership, minimum checks, CI jobs, images, and E2E profiles have one mapping:
  `tools/agent/domains.yaml`.
- Harness code and workflow contracts live in `tools/agent/scripts/**`,
  `tools/ci/**`, `tools/make/agent.mk`, `.pre-commit-config.yaml`, and
  `.github/workflows/**`.
- Contributor prose lives in `AGENTS.md`, `CONTRIBUTING.md`, this directory,
  and the nearest local `AGENTS.md`. It does not duplicate executable routing.

## Local runtime and E2E

- The CLI's local-image flow is the only local runtime path:
  `make vllm-sr-dev`, then `vllm-sr serve --image-pull-policy never` with the
  selected platform.
- E2E profiles live under `e2e/profiles/**` and are run explicitly with
  `make verify PROFILE=<profile>` or selected by the coarse CI registry.
- CI classification lives in `.github/workflows/ci-changes.yml` and
  `tools/ci/classify_pr_changes.py`; `.github/workflows/pr.yml` aggregates its
  selected jobs into the single `PR Gate` check.

## Maintainer operations

- Release, issue, PR, stale-work, and board policy lives in
  `maintainer-ops.md`, `tools/agent/maintainer-policy.yaml`, and the maintainer
  scripts. Generated state stays under `.agent-harness/maintainer/`.
