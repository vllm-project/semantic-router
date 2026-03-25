# Structure Signal Family Implementation Workstream Execution Plan

## Goal

- Add one durable, repo-native workstream for implementing a new `structure` routing signal family that extracts request-shape facts as named signals instead of collapsing them into ad hoc keyword heuristics or projection-only workarounds.
- Keep the implementation aligned across canonical config, router runtime, DSL/platform surfaces, maintained examples, and public docs so the repo exposes one coherent contract for structure-aware routing.
- Close the workstream only after schema, runtime behavior, authoring surfaces, examples, and validations all agree on the same `structure` signal contract.

## Scope

- `docs/agent/plans/**`
- `src/semantic-router/pkg/config/**`
- `src/semantic-router/pkg/classification/**`
- `src/semantic-router/pkg/decision/**`
- `src/semantic-router/pkg/dsl/**`
- `src/semantic-router/pkg/apiserver/**`
- `src/semantic-router/pkg/k8s/**`
- `src/vllm-sr/cli/**`
- `dashboard/frontend/src/**`
- `config/**`
- `deploy/recipes/**`
- `deploy/helm/semantic-router/values.yaml`
- `website/docs/**`
- targeted tests under the touched packages plus maintained asset and docs-contract coverage
- nearest local rules for `src/semantic-router/pkg/config/`, `src/semantic-router/pkg/classification/`, and `src/vllm-sr/cli/`

## Exit Criteria

- The canonical config contract exposes `routing.signals.structure` as a typed family with validated rule shape, normalization/export coverage, and routing-surface catalog support.
- Router runtime evaluates structure rules, records matched rule names plus confidences/metrics, and makes them available to decisions and projections with the same repo-native semantics as other signal families.
- Decision and projection layers can reference `type: structure` without bespoke one-off handling or contract drift.
- DSL authoring, compile/decompile round-trips, JSON export, CLI validation, and any relevant platform/config transport surfaces accept and preserve the new `structure` signal family.
- Maintained examples and public docs explain when to use `structure` versus `keyword`, `context`, and `complexity`, and at least one maintained config asset demonstrates the intended pattern.
- Focused tests and applicable repo-native gates cover schema validation, runtime extraction, DSL round-trip behavior, maintained asset/docs contracts, and any affected platform transport surfaces.

## Task List

- [x] `STR001` Create the indexed execution plan, register it in the plan index, and lock the workstream scope around a dedicated `structure` signal family.
- [x] `STR002` Define the canonical `structure` signal contract in config/schema code, validators, surface catalogs, and canonical import/export helpers.
- [x] `STR003` Implement router runtime extraction, confidence/metrics emission, and signal-usage plumbing for `structure` rules in classification and decision integration paths.
- [x] `STR004` Extend authoring and transport surfaces for `structure`, including DSL/compiler/decompiler/JSON flows, CLI/config helpers, and any affected API, CRD, or dashboard typing seams.
- [x] `STR005` Update maintained configs, examples, and public documentation to show repo-native `structure` usage and clarify its boundary against adjacent signal families.
- [x] `STR006` Run the full validation ladder for the changed-file set, record outcomes in this plan, and promote any durable unresolved contract gap into indexed tech debt.

## Current Loop

- Date: 2026-03-23
- Current task: `STR006` completed
- Branch: `vsr/a035`
- Planned loop order:
  - `L1` lock the execution plan and success criteria
  - `L2` land config/schema and validator support for `routing.signals.structure`
  - `L3` land classification/runtime and decision/projection integration
  - `L4` land DSL/platform/transport surfaces
  - `L5` land maintained examples, docs, and validation closure
- Commands run:
  - startup doc reads for `AGENTS.md`, `.agents/skills/harness/SKILL.md`, `docs/agent/{README.md,plans/README.md,governance.md}`
  - `make agent-report ENV=cpu CHANGED_FILES="docs/agent/plans/README.md docs/agent/plans/pl-0014-structure-signal-family-implementation.md"`
  - broad `codebase-retrieval` across config schema, canonical loaders, classifier runtime, decision integration, DSL/compiler/decompiler, platform transport surfaces, docs/examples, maintained assets, and nearby debt/plan files
  - reference reads for `tools/agent/skills/{harness-contract-change,harness-governance}.md`, `docs/agent/tech-debt/{README.md,td-015-weakly-typed-config-and-dsl-contracts.md,td-018-skill-surface-coverage-drift.md}`, and `docs/agent/plans/pl-0013-openclaw-vsr-install-import-workstream.md`
  - `go test ./pkg/config ./pkg/dsl ./pkg/k8s` (pass)
  - `go test ./pkg/classification ./pkg/services` (source compiled, link blocked by missing native libraries under `candle-binding/target/release` and `nlp-binding/target/release`)
  - `python3 -m compileall src/vllm-sr/cli` (pass)
  - `make dashboard-check` (pass; repo had existing lint warnings only)
  - `make agent-validate` (pass)
  - `make agent-lint CHANGED_FILES="..."` (pass)
  - `make agent-ci-gate CHANGED_FILES="..."` (pass)
  - `make agent-smoke-local` (blocked: local stack not running)
  - `make agent-dev ENV=cpu` (blocked: Docker daemon unavailable at `unix:///Users/bitliu/.docker/run/docker.sock`)

## Decision Log

- 2026-03-23: The workstream will implement a dedicated `structure` signal family, not a generic top-level `feature` family, so the signal catalog stays mechanism-scoped instead of becoming a catch-all for every derived feature.
- 2026-03-23: The intended contract shape is a typed structure detector that can use internal numeric predicates while still emitting named signals to the rest of the routing stack, rather than expanding the repo-wide decision DSL into arbitrary scalar expressions.
- 2026-03-23: `structure` must stay distinct from existing `keyword`, `context`, and `complexity` families; the docs/examples loop must make those boundaries explicit instead of treating `structure` as a second keyword bucket.
- 2026-03-23: The workstream is cross-surface by design, so maintained examples and docs are part of the implementation boundary, not post-hoc cleanup.

## Follow-up Debt / ADR Links

- Potential follow-up debt area only if this workstream proves that named `structure` signals are still insufficient for shared raw numeric features: evaluate whether a later indexed debt item is needed for a generic scalar/feature contract instead of widening this workstream mid-flight.
- No ADR is required yet; this plan is execution-first and should stay task-focused unless the repository makes a durable architecture decision about a general scalar expression layer.
