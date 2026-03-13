# Change Surfaces

This document defines the project-level surfaces used by skills, reports, and validation.

## `router_config_contract`

- Router-side config schema and shared config files consumed directly by the runtime.
- Typical paths: `src/semantic-router/pkg/config/**`, `config/**/*.yaml`
- Task rules: `router-core`, `repo-docs`

## `signal_runtime`

- Signal extraction from request or response content, including heuristic, semantic, or learned text-understanding paths.
- Typical paths: `src/semantic-router/pkg/classification/**`, `req_filter_classification.go`
- Task rules: `router-core`

## `decision_logic`

- Boolean control logic that combines signals and route conditions into decision matches.
- Typical paths: `src/semantic-router/pkg/decision/**`, `req_filter_decision*.go`
- Task rules: `router-core`

## `algorithm_selection`

- Per-decision candidate-model selection after a decision matches.
- Typical paths: `src/semantic-router/pkg/modelselection/**`, `src/semantic-router/pkg/selection/**`, `req_filter_looper*.go`
- Task rules: `router-core`

## `plugin_runtime`

- Post-decision processing such as cache behavior, prompt rewriting, and request or response handling owned by plugins.
- Typical paths: `src/semantic-router/pkg/plugins/**`, `processor_req_body_*.go`, `processor_res_body_*.go`, `req_filter_*.go`
- Task rules: `router-core`

## `native_binding`

- Rust/cgo/onnx/native model bindings used by runtime signals, classifiers, or training artifacts.
- Typical paths: `candle-binding/**`, `ml-binding/**`, `nlp-binding/**`, `onnx-binding/**`
- Task rules: `rust-bindings`, `router-core`

## `response_headers`

- `x-vsr-*` header constants, router emission, dashboard reveal/display allowlists, and user-visible header contracts.
- Typical paths: `src/semantic-router/pkg/headers/**`, `processor_res_header.go`, `HeaderDisplay.tsx`, `HeaderReveal.tsx`
- Task rules: `router-core`, `dashboard`

## `python_cli_schema`

- Python CLI typed schema, parser, validator, merger, and config translation contracts.
- Typical paths: `src/vllm-sr/cli/models.py`, `parser.py`, `validator.py`, `merger.py`
- Task rules: `vllm-sr-cli`

## `python_cli_runtime`

- Python CLI command orchestration, local image management, serve or status flows, and startup wiring.
- Typical paths: `src/vllm-sr/cli/main.py`, `core.py`, `docker_cli.py`, `commands/**`
- Task rules: `vllm-sr-cli`

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
- Typical paths: `PlaygroundPage.tsx`, `ChatComponent*.tsx`, `HeaderDisplay.tsx`, `HeaderReveal.tsx`
- Task rules: `dashboard`

## `dsl_crd`

- DSL compiler/decompiler and translation layers that bridge router config to Kubernetes-facing forms.
- Typical paths: `src/semantic-router/pkg/dsl/**`, `src/semantic-router/pkg/k8s/**`
- Task rules: `router-core`, `operator-stack`, `e2e-framework`

## `k8s_operator`

- Operator APIs, CRDs, controller-facing config translation, and Kubernetes deployment control-plane behavior.
- Typical paths: `deploy/operator/**`, `deploy/kubernetes/crds/**`, `src/semantic-router/pkg/apis/**`
- Task rules: `operator-stack`, `e2e-framework`

## `training_post_training`

- Post-training, classifier fine-tuning, model-classifier data pipelines, and training artifacts that feed runtime behavior.
- Typical paths: `src/training/**`, `tools/make/models.mk`, `scripts/train-mmbert32k-gpu.sh`, `website/docs/training/**`
- Task rules: `training-stack`, `repo-docs`

## `docs_examples`

- User-facing docs, examples, presets, and reference configs.
- Typical paths: `docs/**`, `website/**`, `deploy/amd/**`, `config/**/*.yaml`
- Task rules: `repo-docs`, `training-stack`

## `harness_docs`

- Shared agent entry, indexed harness docs, local `AGENTS.md` supplements, skill prose, ADRs, execution plans, glossary, and debt tracking for the harness itself.
- Typical paths: `AGENTS.md`, `docs/agent/**`, `tools/agent/skills/**`, indexed local `AGENTS.md` files under hotspot directories
- Task rules: `agent_text`, `repo-docs`

## `harness_exec`

- Executable harness manifests, scripts, Make entrypoints, and validation logic that implement the shared contract.
- Typical paths: `tools/agent/*.yaml`, `tools/agent/scripts/**`, `tools/make/agent.mk`
- Task rules: `agent_exec`

## `contributor_interface`

- Contributor-facing wrappers around the harness such as README, contributing guidance, PR or issue intake templates, and maintainer label taxonomy.
- Typical paths: `README.md`, `CONTRIBUTING.md`, `.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/**`, `.prowlabels.yaml`
- Task rules: `repo-docs`, `agent_text`

## `local_smoke`

- Canonical local image build, serve, dashboard/router smoke validation, and environment-specific smoke configs.
- Typical paths: `tools/make/agent.mk`, `tools/make/docker.mk`, `src/vllm-sr/cli/**`, `config/testing/config.agent-smoke.*.yaml`
- Task rules: `vllm-sr-cli`

## `local_e2e`

- Affected local E2E profile selection and local profile execution.
- Typical paths: `tools/agent/e2e-profile-map.yaml`, `e2e/profiles/**`, `config/testing/**`, `deploy/kubernetes/**`
- Task rules: `e2e-framework`

## `ci_e2e`

- CI fanout, change classification, and standard profile-matrix execution for the merge gate.
- Typical paths: `.github/workflows/ci-changes.yml`, `integration-test-k8s.yml`, `integration-test-vllm-sr-cli.yml`, `pre-commit.yml`, `tools/agent/skill-registry.yaml`
- Task rules: `agent_exec`, `e2e-framework`
