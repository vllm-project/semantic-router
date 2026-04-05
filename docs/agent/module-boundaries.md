# Module Boundaries

This document describes the repository's major subsystem seams and the boundaries agents should preserve while editing.

## Top-Level Subsystems

- `src/semantic-router/`
  - owns router runtime, config loading, decision logic, extproc processing, and controller-side integration
- `src/vllm-sr/`
  - owns the Python CLI, config generation, local startup orchestration, and Docker-facing developer flow
- `candle-binding/`, `ml-binding/`, `nlp-binding/`, `onnx-binding/`
  - own native model and classifier bindings consumed by router-side runtime paths
- `dashboard/`
  - owns frontend and backend UI surfaces, topology visualization, and playground/reveal presentation
- `deploy/operator/` and `src/semantic-router/pkg/apis/`
  - own Kubernetes operator contracts, CRD types, and cluster-facing deployment schema
- `src/training/`
  - owns post-training, classifier fine-tuning, and training/evaluation workflows for router models
- `e2e/`
  - owns local and CI profile validation, E2E driver code, and deploy-profile coverage
- `tools/agent/`
  - owns the agent harness manifests, scripts, skills, and structure rules

## Boundary Rules

- Production code must not depend on `e2e/`, `docs/agent/`, `tools/agent/`, or website content.
- Dashboard backend must not depend on dashboard frontend source.
- Local CLI runtime behavior belongs in `src/vllm-sr/`; router runtime behavior belongs in `src/semantic-router/`.
- Control-plane surfaces such as the CLI, dashboard backend, and operator should depend on versioned router contracts or explicit public runtime-service seams instead of deep router-runtime internals when the dependency crosses a product boundary.
- Native bindings stay behind runtime seams instead of leaking binding-specific setup across the codebase.
- Dashboard backend handler files should separate HTTP transport from config persistence, deploy/rollback control, and runtime status collection.
- Dashboard frontend should separate route-shell/auth gating from page-specific orchestration and from large chat or editor containers.
- Page-level dashboard files should orchestrate route state, while support types, builders, and presentational fragments live in adjacent modules.
- Router config package files should separate schema declaration, canonical conversion/export, plugin-family contracts, and semantic validation.
- Classification runtime should separate model discovery/bootstrap, per-family inference, and service assembly instead of coupling them through one hotspot orchestrator.
- Shared backend lifecycle policy belongs in reusable seams; domain packages should not each recreate connection, bootstrap, and retry logic for the same store technology.
- E2E testcase files should own one externally visible contract or one explicitly named benchmark concern, not both.
- Fleet-sim optimizer code should separate analytical sizing, simulation verification, and public export policy instead of widening one model-and-reporting hotspot.
- Operator API and controller code should separate CRD schema declaration, webhook validation, controller-side canonical translation, and sample or generated contract upkeep.
- Extproc response code should separate provider normalization, streaming accumulation/finalization, replay or cache persistence, and response-side warning shaping.

## Core Capability Placement

- Put text-derived extraction capabilities in the `signal` layer.
  - If a new feature extracts information from request or response text through heuristics, semantic rules, or learned models, it belongs with signals.
  - Examples: context heuristics, jailbreak detection, semantic classification from text.
- Put boolean control logic in the `decision` layer.
  - If a new feature combines one or more signals, thresholds, or route conditions and the work is primarily about boolean matching, gating, or control flow, it belongs with decisions.
  - Examples: combining keyword and classifier signals with AND/OR logic, route guards, safety gates, decision priority handling.
- Put per-decision multi-model selection in the `algorithm` layer.
  - If a feature starts after a decision has matched and its purpose is to choose among multiple candidate models for that decision, it belongs with algorithms.
  - Examples: latency-aware selection, cost-aware selection, weighted model ranking inside a decision's `modelRefs`.
- Put decision-driven processing in the `plugin` layer.
  - If a new feature acts on a decision or algorithm result or transforms request/response data after routing logic decides something, it belongs with plugins.
  - Examples: cache behavior, prompt rewriting, post-decision request/response processing.
- Put cross-cutting behavior in the global level.
  - If a capability should apply across the whole router or configuration model and is not primarily owned by signal extraction, decision control logic, algorithmic model choice, or plugin-side processing, it belongs in the global layer.
- Decide placement before implementation.
  - Do not hide signal-style extraction logic inside decisions, algorithms, or plugins.
  - Do not hide decision-style boolean control logic inside signals or plugins.
  - Do not bury multi-model selection heuristics inside signals, decisions, or global config.
  - Do not put plugin-style request/response processing into global config just because it affects many routes.
  - When a feature spans more than one layer, define the primary owner first and keep the other layers as adapters, config plumbing, or presentation only.

## Hotspot-Specific Local Rules

Before editing these areas, read the nearest local `AGENTS.md`:

- `src/semantic-router/pkg/config/`
- `src/semantic-router/pkg/classification/`
- `src/semantic-router/pkg/extproc/`
- `src/vllm-sr/cli/`
- `src/fleet-sim/fleet_sim/optimizer/`
- `deploy/operator/api/v1alpha1/`
- `deploy/operator/controllers/`
- `dashboard/frontend/src/`
- `dashboard/frontend/src/pages/`
- `dashboard/frontend/src/components/`
- `dashboard/backend/handlers/`
- `e2e/testcases/`

See [local-rules.md](local-rules.md) for the indexed list of local harness supplements.

## Source of Truth

- Human-readable overview: [repo-map.md](repo-map.md)
- Structural policy: [architecture-guardrails.md](architecture-guardrails.md)
- Executable dependency checks: [../../tools/agent/structure-rules.yaml](../../tools/agent/structure-rules.yaml)
