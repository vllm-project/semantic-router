# Change Surfaces

This document defines the project-level surfaces used by agent skills and reports.

## `router_config_contract`

- Router-side config schema, DSL emission, CRD conversion, and example config compatibility.
- Typical paths: `src/semantic-router/pkg/config/**`, `src/semantic-router/pkg/dsl/**`, `src/semantic-router/pkg/k8s/**`, `config/**/*.yaml`
- Task rules: `router-core`, `repo-docs`

## `router_runtime`

- Signal evaluation, decision execution, extproc handling, and runtime model-selection behavior.
- Typical paths: `src/semantic-router/pkg/classification/**`, `src/semantic-router/pkg/decision/**`, `src/semantic-router/pkg/extproc/**`, `src/semantic-router/pkg/modelselection/**`
- Task rules: `router-core`

## `native_binding`

- Rust/cgo/onnx/native model bindings used by runtime classifiers or signal evaluation.
- Typical paths: `candle-binding/**`, `ml-binding/**`, `nlp-binding/**`, `onnx-binding/**`
- Task rules: `rust-bindings`, `router-core`

## `response_headers`

- `x-vsr-*` header constants, router emission, dashboard display allowlists, and reveal overlays.
- Typical paths: `src/semantic-router/pkg/headers/**`, `processor_res_header.go`, `HeaderDisplay.tsx`, `HeaderReveal.tsx`
- Task rules: `router-core`, `dashboard`

## `python_cli_schema`

- Python CLI typed schema, parser, validator, merger, and config translation.
- Typical paths: `src/vllm-sr/cli/models.py`, `parser.py`, `validator.py`, `merger.py`
- Task rules: `vllm-sr-cli`

## `dashboard_config_ui`

- Dashboard config editor, config display, and UI mutation flows.
- Typical paths: `dashboard/frontend/src/pages/ConfigPage.tsx`, `BuilderPage.tsx`, `DslEditorPage.tsx`
- Task rules: `dashboard`

## `topology_visualization`

- Topology parsing, layout, graph nodes, and highlighted decision-path visualization.
- Typical paths: `dashboard/frontend/src/pages/topology/**`, `dashboard/backend/handlers/topology.go`
- Task rules: `dashboard`

## `playground_reveal`

- Playground chat response reveal, header display, and user-visible signal presentation.
- Typical paths: `PlaygroundPage.tsx`, `ChatComponent.tsx`, `HeaderDisplay.tsx`, `HeaderReveal.tsx`
- Task rules: `dashboard`

## `dsl_crd`

- DSL compiler/decompiler, CRD conversion, and Kubernetes-facing route translation.
- Typical paths: `src/semantic-router/pkg/dsl/**`, `src/semantic-router/pkg/k8s/**`, `deploy/operator/**`
- Task rules: `router-core`, `e2e-framework`

## `docs_examples`

- User-facing docs, examples, presets, and reference configs.
- Typical paths: `docs/**`, `website/**`, `deploy/amd/**`, `config/**/*.yaml`
- Task rules: `repo-docs`

## `harness_docs`

- Shared agent entry, indexed harness docs, local `AGENTS.md` supplements, and skill prose for the repository contract itself.
- Typical paths: `AGENTS.md`, `docs/agent/**`, `tools/agent/skills/**`, indexed local `AGENTS.md` files under hotspot directories
- Task rules: `agent_text`, `repo-docs`

## `harness_exec`

- Executable harness manifests, scripts, and Make entrypoints that implement the contract.
- Typical paths: `tools/agent/*.yaml`, `tools/agent/scripts/**`, `tools/make/agent.mk`
- Task rules: `agent_exec`, `make-and-ci`

## `contributor_interface`

- Contributor-facing wrappers that expose the harness through README, contributing guidance, PR templates, and issue intake forms.
- Typical paths: `README.md`, `CONTRIBUTING.md`, `.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/**`
- Task rules: `repo-docs`, `agent_text`

## `local_smoke`

- Canonical local image build, serve, and smoke validation path.
- Typical paths: `tools/make/agent.mk`, `tools/make/docker.mk`, `src/vllm-sr/cli/**`, `config/testing/config.agent-smoke.*.yaml`
- Task rules: `vllm-sr-cli`, `make-and-ci`

## `local_e2e`

- Affected local E2E profile selection and local profile execution.
- Typical paths: `tools/agent/e2e-profile-map.yaml`, `e2e/profiles/**`, `config/testing/**`, `deploy/kubernetes/**`
- Task rules: `e2e-framework`

## `ci_e2e`

- CI fanout, change classification, and standard profile matrix execution.
- Typical paths: `.github/workflows/ci-changes.yml`, `integration-test-k8s.yml`, `pre-commit.yml`, `tools/agent/skill-registry.yaml`
- Task rules: `agent-exec`, `ci-general`, `e2e-framework`
