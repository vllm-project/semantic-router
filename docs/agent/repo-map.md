# Repo Map

## Core Subsystems

- `src/semantic-router/`
  - Go router, Envoy extproc server, config loading, Kubernetes controller, routing logic
- `src/vllm-sr/`
  - Python CLI, config generation, Docker orchestration, local bootstrap flow
- `src/fleet-sim/`
  - Fleet simulator package, HTTP service, data sets, examples, and service image used by the dashboard sidecar
- `candle-binding/`, `ml-binding/`, `nlp-binding/`
  - Rust-backed inference and ML bindings used by the router
- `dashboard/`
  - Dashboard frontend and backend
- `deploy/operator/`
  - Kubernetes operator, CRD schema, and controller-facing deployment contract
- `src/training/`
  - classifier/post-training scripts, training utilities, and evaluation-oriented model workflows
- `e2e/`
  - kind/Kubernetes E2E framework and profile matrix
- `tools/make/`
  - canonical automation entry points
- `tools/agent/`
  - coding-agent manifests, scripts, skills, and structure rules

## Known Hotspots

- `src/semantic-router/pkg/config/config.go`
  - central config type table; prefer moving plugin contracts and helper methods into adjacent files
- `src/semantic-router/pkg/config/validator.go`
  - semantic validation hotspot; keep family-specific checks and inventories in sibling validator files instead of widening one entrypoint
- `src/semantic-router/pkg/config/rag_plugin.go`
  - plugin-config hotspot; keep backend-specific RAG payloads, decode helpers, and validation branches in adjacent modules as the matrix grows
- `src/semantic-router/pkg/classification/classifier.go`
  - classification orchestration hotspot; keep discovery, family adapters, mappings, and runtime execution on separate seams
- `src/semantic-router/pkg/extproc/processor_req_body.go`
  - request-body orchestration hotspot; prefer extracting narrow helpers instead of extending the main flow
- `src/semantic-router/pkg/extproc/processor_res_body.go`
  - response-body orchestration hotspot; keep provider normalization, streaming, replay/cache, and response-warning helpers separate
- `src/fleet-sim/fleet_sim/optimizer/base.py`
  - fleet-sim optimizer hotspot; keep analytical sizing, DES verification, and report-oriented helpers on separate seams
- `deploy/operator/api/v1alpha1/semanticrouter_types.go`
  - operator schema hotspot; keep CRD contract families and comments readable instead of widening one type table
- `deploy/operator/controllers/canonical_config_builder.go`
  - operator canonical-config hotspot; keep family-specific translation and backend discovery out of one builder seam
- `dashboard/frontend/src/App.tsx`
  - dashboard route-shell hotspot; keep auth/setup gating and layout composition separate from page-specific logic
- `dashboard/frontend/src/pages/DashboardPage.tsx`
  - dashboard overview hotspot; keep fetch shaping, quick-action composition, and render fragments out of one page shell
- `e2e/testcases/`
  - acceptance coverage hotspot; keep contract assertions separate from report-only metrics and stress dashboards
- `dashboard/frontend/src/pages/ConfigPage.tsx`
  - dashboard config editor hotspot; keep schema/types/helpers outside the page component when possible
- `dashboard/frontend/src/pages/SetupWizardPage.tsx`
  - setup route hotspot; keep route state in the page but move wizard support types, config builders, and step panels into adjacent modules
- `dashboard/frontend/src/components/ChatComponent.tsx`
  - playground/chat orchestration hotspot; keep cards, citation rendering, and control widgets in adjacent components
- `dashboard/frontend/src/components/ExpressionBuilder.tsx`
  - expression editor hotspot; keep AST/tree helpers and reusable render fragments outside the main ReactFlow container
- `dashboard/backend/handlers/config.go`, `dashboard/backend/handlers/deploy.go`, `dashboard/backend/handlers/status.go`, `dashboard/backend/handlers/status_modes.go`
  - dashboard backend control hotspots; keep HTTP transport thin and move file mutation, runtime apply, container probing, and status synthesis into adjacent helpers

## Main Entry Points

- Local image build: [tools/make/docker.mk](../../tools/make/docker.mk)
- Root command router: [Makefile](../../Makefile)
- Local serve command: [src/vllm-sr/cli/main.py](../../src/vllm-sr/cli/main.py)
- Runtime command orchestration: [src/vllm-sr/cli/commands/runtime.py](../../src/vllm-sr/cli/commands/runtime.py)
- Local service startup/status: [src/vllm-sr/cli/core.py](../../src/vllm-sr/cli/core.py)
- Docker compatibility barrel and image/service helpers: [src/vllm-sr/cli/docker_cli.py](../../src/vllm-sr/cli/docker_cli.py)
- E2E driver: [tools/make/e2e.mk](../../tools/make/e2e.mk)

## High-Risk Areas

- `src/vllm-sr/**`
  - affects local developer experience and local image startup
- `src/fleet-sim/**`
  - affects simulator packaging, dashboard fleet-planning workflows, and local sidecar startup
- `deploy/operator/**`
  - affects CRD schema, generated manifests, and controller-side config translation
- `src/semantic-router/**`
  - affects router behavior and broad E2E coverage
- `tools/make/**`
  - affects shared developer and CI entry points
- `e2e/pkg/**`, `e2e/cmd/**`, `e2e/testcases/**`
  - affects all E2E profiles
- `tools/agent/**`, `AGENTS.md`
  - affects agent workflow and validation behavior

## Recommended First Reads

- [docs/agent/README.md](README.md)
- [AGENTS.md](../../AGENTS.md)
- [docs/agent/governance.md](governance.md)
- [docs/agent/tech-debt-register.md](tech-debt-register.md)
- [docs/agent/tech-debt/README.md](tech-debt/README.md)
- [docs/agent/change-surfaces.md](change-surfaces.md)
- [docs/agent/environments.md](environments.md)
- [docs/agent/feature-complete-checklist.md](feature-complete-checklist.md)
- [tools/agent/skill-registry.yaml](../../tools/agent/skill-registry.yaml)
- [tools/agent/task-matrix.yaml](../../tools/agent/task-matrix.yaml)
- [tools/agent/e2e-profile-map.yaml](../../tools/agent/e2e-profile-map.yaml)
