---
title: MoM First-Class Evaluation
description: Proposed versioned contract, ownership boundaries, artifact placement, and delivery plan for evaluating published Mixtures-of-Models as complete system identities.
created: 2026-08-28
status: Proposal
---

> **Status:** Proposal · **Created:** 2026-08-28 · **Epic:** [#3038](https://github.com/vllm-project/semantic-router/issues/3038)

## Problem

Every published Mixture-of-Models (MoM) is a first-class model identity from the user's
perspective, but today we lack a versioned, reproducible contract for evaluating that
complete system against qualified standalone baselines and prior releases.

Decision-level routing evaluation (`probes.yaml`, `vllm-sr eval`) answers whether an
individual routing decision was correct. It does not answer whether the complete MoM
delivers a better end-to-end model experience across quality, safety, latency, cost,
and operational behavior.

## Proposal

Introduce a versioned MoM evaluation contract (`vllm-sr/mom-evaluation/v1`) with:

1. A per-recipe **evaluation manifest** that names entrypoints, baselines, and runtime
   equivalence rules.
2. A mandatory **core suite** every published MoM must pass before publication.
3. An **extension pack registry** for objective-specific proof without weakening the
   core suite.
4. A **result bundle schema** that records identity, metrics, baselines, regression,
   provenance, and publication classification.
5. A **delivery plan** split into bounded, assigned child issues after this proposal is
   agreed.

**This PR is proposal-only.** It documents the contract design, artifact ownership,
versioning rules, and delivery slices. It does not land schemas, manifests, runners,
scorecards, CI workflows, or recipe publication integration. Contract artifacts land
in assigned child issues — starting with [#3238](https://github.com/vllm-project/semantic-router/issues/3238)
(MOM-01) — after this proposal is approved.

## Evaluation contract layers

| Layer | Artifact | Role |
| --- | --- | --- |
| Recipe manifest | `mom-evaluation.yaml` per MoM family bundle or standalone MoM recipe | Names entrypoints, baselines, extension packs, and runtime defaults for a published MoM identity. |
| Core suite | Versioned core-suite manifest | Mandatory benchmarks and operational metrics for every MoM. |
| Baseline protocol | Versioned baseline-protocol document | Standalone comparison modes and equivalence rules. |
| Extension packs | Versioned pack manifests | Objective-specific additive metrics with graduation criteria. |
| Pack registry | Registry document | Stable pack IDs, manifest references, and evaluator binding convention. |
| Result bundle | `mom_eval_result.json` per run | Machine-readable output with provenance and publication gate. |

### Complementary evaluation surfaces

| Question | Surface | Owner |
| --- | --- | --- |
| Did routing select the expected algorithm or model? | `probes.yaml`, `vllm-sr eval` | MoM & Routing / decision-level eval (#2333) |
| Does the complete MoM beat a qualified standalone model? | MoM evaluation (this proposal) | Evaluation & Quality |
| Did a code path regress allocations or latency? | `perf/` harnesses | Data Plane & Networking |

Extension pack regressions never override a core-suite failure.

## Artifact ownership, location, and versioning

Contract artifacts must not land until a child issue is assigned. The table below
settles **who owns each artifact**, **where it should live**, and **how it is
versioned** so implementation PRs do not invent parallel locations.

| Artifact | Owner | Proposed location | Versioning |
| --- | --- | --- | --- |
| Per-recipe evaluation manifest | MoM & Routing (content) · Evaluation & Quality (schema) | Authoring: `config/recipes/built-in/<channel>/<bundle>/mom-evaluation.yaml` for built-in MoM families; `config/recipes/<name>/mom-evaluation.yaml` for standalone maintained MoM recipes. Package mirror: `src/vllm-sr/cli/model_assets/<channel>/<bundle>/mom-evaluation.yaml` (generated only) | `schema_version: vllm-sr/mom-evaluation/v1` |
| Manifest and result JSON Schemas | Evaluation & Quality | `config/evaluation/schema/` | `$id` under `https://vllm-sr.ai/schemas/`; bump only via new schema version |
| Core suite manifest | Evaluation & Quality | `config/evaluation/mom-core-suite/v1/manifest.yaml` | `schema_version: vllm-sr/mom-core-suite/v1`; suite `version` semver |
| Baseline protocol | Evaluation & Quality | `config/evaluation/baseline-protocol/v1.yaml` | `schema_version: vllm-sr/baseline-protocol/v1` |
| Extension pack manifests | Evaluation & Quality · objective workgroups (content) | `config/evaluation/packs/<name>/v1/manifest.yaml` | `schema_version: vllm-sr/evaluation-pack/v1`; pack `id` + semver |
| Pack registry | Evaluation & Quality | `config/evaluation/packs/registry.yaml` | `schema_version: vllm-sr/evaluation-pack-registry/v1` |
| Published scorecards | Evaluation & Quality | `config/evaluation/scorecards/<family>/<entrypoint>/<version>/` | Immutable per release; index for regression lookup |
| Runner, collectors, publish tooling | Evaluation & Quality | `tools/evaluation/mom/` (new tree under `tools/`, not a new `bench/` root) | Package version aligned with contract semver |
| Make targets and CI workflow | Evaluation & Quality | `tools/make/mom-eval.mk`, `.github/workflows/mom-eval-rc.yml` | Gate on assigned MOM-04 slice |
| Reference example manifest | Evaluation & Quality | `config/evaluation/examples/` (not wired into recipe dirs until MOM-02) | Illustrative only; excluded from recipe conformance until assigned |

Versioning rules:

1. **Contract-first** — schema `$id` values and `schema_version` fields are the public
   API. Implementation may lag, but must not publish alternate paths or thresholds
   outside the assigned contract slice.
2. **Immutable published scores** — once a scorecard is published for an entrypoint
   release, its result bundle and metric values are retained for regression comparison.
3. **No premature source of truth** — blocking thresholds, baseline model arms, harness
   bindings, and evaluator modules land only when the owning child issue is assigned
   and a validator or runner consumes them.
4. **Reuse existing harnesses** — core quality layers should prefer existing EvalScope
   and `bench/router_flow/real_eval/` integrations where possible; new harness code
   belongs under `tools/evaluation/mom/`, not a new top-level `bench/mom_eval/` tree.

### Per-recipe manifest source of truth and synchronization

The per-MoM evaluation manifest is **recipe- or bundle-local**, not a standalone
catalog fragment under `config/evaluation/`. Shared suite, baseline, and pack
artifacts remain in `config/evaluation/` and are referenced by path from the
manifest (`core_suite_ref`, `baseline_protocol_ref`, `extension_packs`). The
manifest itself is owned with the MoM identity it evaluates.

| MoM publication surface | Authoring source of truth | Package mirror |
| --- | --- | --- |
| Built-in catalog asset (for example `mom-v1`) | `config/recipes/built-in/<channel>/<bundle>/mom-evaluation.yaml` | `src/vllm-sr/cli/model_assets/<channel>/<bundle>/mom-evaluation.yaml` |
| Standalone maintained MoM recipe | `config/recipes/<name>/mom-evaluation.yaml` | None until the recipe is packaged |

Settled rules:

1. **One manifest per MoM family bundle or standalone MoM recipe** — a built-in
   catalog asset such as `mom-v1` carries one family-level manifest that names
   every catalog entrypoint for that asset. Standalone MoM recipes carry their
   own manifest alongside the existing five-file recipe contract.
2. **No catalog-indirection authoring** — `catalog.yaml` does not reference a
   separate evaluation fragment. Catalog entries continue to bind models to an
   `asset` bundle; evaluation content lives in that bundle directory.
3. **CLI copy is generated, not authored** — the installable package mirror under
   `src/vllm-sr/cli/model_assets/` follows the same rule as `catalog.yaml` and
   bundled recipe files: maintainers edit only the authoring tree under
   `config/recipes/built-in/`, then regenerate the mirror with
   `tools/release/sync_model_catalog.py` (extended in MOM-02 to include
   `mom-evaluation.yaml`).
4. **Drift is rejected in CI** — the Built-in Model Catalog workflow runs
   `sync_model_catalog.py --check`; direct edits to `cli/model_assets/` fail the
   gate, matching the existing built-in catalog source/package parity rule enforced
   by `tools/release/sync_model_catalog.py`.
5. **Runners resolve the authoring path** — validators and runners take an explicit
   manifest path from the caller (recipe directory, bundle path, or generated
   runtime materialization). Formal runs record the resolved authoring path and
   bundle or recipe digest in result provenance; the reference example under
   `config/evaluation/examples/` remains illustrative and is never treated as a
   publication source.

## Schemas

Two JSON Schemas define the steady-state contract (to land in MOM-01 / #3238):

| Schema | `$id` | Purpose |
| --- | --- | --- |
| MoM evaluation manifest | `vllm-sr/mom-evaluation/v1` | Validates per-recipe `mom-evaluation.yaml` documents. |
| MoM eval result bundle | `vllm-sr/mom-eval-result/v1` | Validates publishable run output and regression summaries. |

### Manifest shape (summary)

```yaml
schema_version: vllm-sr/mom-evaluation/v1
mom:
  recipe_id: multi-objective
  recipe_version: 0.1.0
core_suite_ref: <versioned core-suite manifest path>
baseline_protocol_ref: <versioned baseline-protocol path>
runtime:
  api_url: http://127.0.0.1:8899/v1
  generation:
    temperature: 0
    max_tokens: 8192
  concurrency: 4
  timeout_seconds: 900
entrypoints:
  vllm-sr/mom-v1-blend:
    objective: general-purpose
    extension_packs: []
    baselines:
      - model: local/qwen3.6-27b-coder
        match: qualified_standalone
        role: balanced_standalone
```

Required manifest fields: `schema_version`, `mom`, `core_suite_ref`, `entrypoints`,
`runtime`. Optional: `baseline_protocol_ref`, `provenance`.

Each entrypoint declares an `objective`, zero or more `extension_packs`, and at least
one baseline arm with `model`, `match` (`matched_compute`, `qualified_standalone`, or
`fixed_budget`), and `role`.

### Result bundle shape (summary)

Required top-level sections: `identity`, `contract`, `environment`, `metrics`,
`baselines`, `publication`. Optional: `regression`, `diagnostics`, `artifacts`.

`identity.run_mode` is one of `smoke`, `formal`, or `release-candidate`.
`publication.publishable` is `true` only when provenance, core blocking metrics, and
run mode requirements are satisfied.

## Core suite v1 (proposed)

The mandatory core suite covers six layers:

| Layer | Blocking | Examples |
| --- | --- | --- |
| General quality | Yes | GPQA-D, MMLU-Pro |
| Instruction following | Yes | IFEval |
| Robustness | No (diagnostic) | Request-class consistency matrix |
| Safety baseline | Yes | Harmful-request containment |
| Operational | Partial | p50/p99 latency, token usage, failure rate |
| Regression | Yes | Delta vs previous released scorecard |

Proposed publication rules:

- All blocking core metrics must pass.
- Extension packs cannot override a core failure.
- Minimum run mode for publication: `formal`.
- `smoke` runs are classified as diagnostic only.
- Operational `failure_rate` above 10% blocks publication even when quality metrics pass.

Core quality benchmarks should integrate with the existing EvalScope suite reference
(`bench/router_flow/real_eval/evalscope_suite.yaml`) where adapters exist.

## Baseline protocol v1 (proposed)

Standalone-model comparison modes:

| Mode | Use |
| --- | --- |
| `matched_compute` | Same token budget or primary pool member for uplift diagnostics. |
| `qualified_standalone` | Best single-model baseline for the entrypoint objective. |
| `fixed_budget` | Cost-capped comparison (reserved for cost packs). |

Equivalence rules require the same dataset revision, generation config, concurrency,
timeout, and recorded baseline arms in the result bundle.

Initial entrypoint baseline mapping (subject to maintainer hardware and pool updates
when MOM-01 lands):

| Entrypoint | Primary baseline role | Extension packs |
| --- | --- | --- |
| `vllm-sr/mom-v1-blend` | balanced_standalone | — |
| `vllm-sr/mom-v1-lite` | economy_standalone | cost/v1 |
| `vllm-sr/mom-v1-flash` | latency_standalone | latency/v1 |
| `vllm-sr/mom-v1-ultra` | quality_standalone | orchestration/v1 |
| `vllm-sr/mom-v1-vault` | security_standalone | security/v1 |

## Extension packs (proposed v1 set)

| Pack | Objective | Entrypoint | Key metrics |
| --- | --- | --- | --- |
| `cost/v1` | cost-optimized | `vllm-sr/mom-v1-lite` | quality at fixed cost, budget adherence |
| `latency/v1` | latency-focused | `vllm-sr/mom-v1-flash` | time to first token, p99 latency |
| `security/v1` | security-first | `vllm-sr/mom-v1-vault` | jailbreak containment, PII handling |
| `orchestration/v1` | quality-orchestration | `vllm-sr/mom-v1-ultra` | orchestration quality delta, call budget adherence |

Each pack manifest declares datasets, metrics, graduation criteria, baseline role, and
`blocking_rules.cannot_override_core_suite: true`. Evaluator binding convention
(Python entry points vs documented module path) is an open question resolved before
MOM-03 lands.

## Ownership boundaries

| Workgroup / surface | Owns | Does not own |
| --- | --- | --- |
| **Evaluation & Quality** | Core suite versioning, baseline protocol, pack registry, result schema, publication rules, scorecard retention policy, runner under `tools/evaluation/mom/` | Router recipe semantics, model pool membership |
| **MoM & Routing** (#2971, #2238) | MoM identity, entrypoints, recipe lifecycle, model-card fields that reference evaluation | Benchmark harness selection beyond routing probes |
| **Router Models & Inference Runtime** | Router Model artifacts and isolated model evaluation | End-to-end MoM score publication |
| **Data Plane & Networking** | Request-path latency, failure, and resource measurements consumed by operational metrics | Evaluation contract authorship |
| **Enterprise & Environment** | Release and rollback gates that consume scorecards | Benchmark execution |
| **Objective workgroups** (Agentic, Security, etc.) | New extension pack manifests and graduation criteria for their objectives | Core suite weakening |

Recipe conformance checks for MoM manifests belong in the agent harness once MOM-02
validation tooling is implemented.

## Reproducibility and provenance

A score is publishable only when the run identity is sufficient to reproduce the
evaluation. The result bundle must record:

| Field group | Required content |
| --- | --- |
| **MoM identity** | Recipe ID and version, entrypoint, objective, recipe digest, pool membership |
| **Contract** | Core suite version, declared extension pack IDs and versions |
| **Environment** | Platform, router image, dataset revisions, generation config, API URL |
| **Baselines** | Each standalone arm with role, match mode, and per-metric results |
| **Command & artifacts** | Invoked command, raw benchmark outputs, regression report paths |

Provenance rules:

1. **Recipe snapshot** — evaluation runs against the exact recipe revision named in
   `identity.recipe_digest`.
2. **Dataset pinning** — `environment.dataset_revisions` records benchmark dataset
   versions used by each layer.
3. **Generation equivalence** — baseline arms use the same `runtime.generation` as
   the MoM under test.
4. **Run mode classification** — smoke results are never publishable as launch scores.
5. **Historical retention** — published scorecards are retained under the scorecard
   path above with an index for regression lookup (MOM-05 slice).

## Delivery slices

After this proposal is agreed, #3038 splits into formally assigned child issues.
Implementation must not proceed from the Epic directly. Paused work on
`feat/mom-first-class-evaluation-3038` / [#3083](https://github.com/vllm-project/semantic-router/pull/3083)
reuses against these slices — not against the Epic.

| Slice | Child issue | Scope | Lands |
| --- | --- | --- | --- |
| **MOM-01** | [#3238](https://github.com/vllm-project/semantic-router/issues/3238) | JSON Schemas, core suite manifest, baseline protocol, pack registry and pack manifests | `config/evaluation/**` contract tree |
| **MOM-02** | TBD | Static validation tooling, recipe conformance integration for published MoM recipes, and `sync_model_catalog.py` extension for `mom-evaluation.yaml` mirror checks | `make mom-eval-validate`, agent harness checks, Built-in Model Catalog workflow |
| **MOM-03** | TBD | Runner, collectors, regression comparison, failure slicing, publish pipeline | `tools/evaluation/mom/` |
| **MOM-04** | TBD | Make targets and smoke CI workflow | `tools/make/mom-eval.mk`, `.github/workflows/mom-eval-rc.yml` |
| **MOM-05** | TBD | Reference scorecards for all five MoM V1 entrypoints | `config/evaluation/scorecards/` |
| **MOM-06** | TBD | User guide, maintainer skill, execution plan index entry | `website/docs/benchmarking/`, agent skill |
| **MOM-07** | TBD | Model-card and catalog integration; formal runs on maintainer hardware | Model cards, catalog metadata |

## Validation plan

Validation is split into static contract checks (MOM-02) and runtime gates (MOM-03+).
This proposal defines the target checks only.

### Static validation (MOM-02)

| Check | Input | Rule |
| --- | --- | --- |
| Manifest schema | Authoring `mom-evaluation.yaml` in built-in MoM bundles or standalone MoM recipes | Validates against manifest JSON Schema |
| Core suite reference | Manifest `core_suite_ref` | Referenced file exists; version matches `vllm-sr/mom-core-suite/v1` |
| Baseline protocol reference | Manifest `baseline_protocol_ref` | Referenced file exists; entrypoint baselines align with protocol |
| Pack registry | Declared `extension_packs` | Each pack ID resolves in the pack registry |
| Pack manifests | Per-pack YAML | Metrics declare blocking classification and graduation thresholds |
| Result schema | Sample or generated bundles | Validates against result JSON Schema |

Proposed gate: `make mom-eval-validate` (not shipped in this proposal PR).

Integration with existing gates:

- `make recipe-conformance-static` gains MoM manifest presence checks for published
  MoM recipes once MOM-02 lands.
- Decision-level probes remain mandatory and independent.

### Runtime validation (MOM-03 onward)

| Mode | Purpose | Publication |
| --- | --- | --- |
| `smoke` | Local diagnostic with reduced sample limits | Never publishable |
| `formal` | Full sample limits on maintainer hardware | Publishable when core suite passes |
| `release-candidate` | Formal run plus regression vs previous scorecard | Required before model-card update |

Regression blocking uses per-benchmark thresholds in the core suite manifest and
pack graduation criteria. Core-suite failures block publication regardless of extension
pack scores.

## Non-goals

- Replacing decision-level routing evaluation (#2333).
- Evaluating Router Models in isolation.
- Hosted benchmark SLA or always-on formal runs in CI without live backends.
- Landing executable contract files, runnable harnesses, synthetic scorecards, or
  model-card updates in this proposal PR.

## Open questions

1. Which maintainer-owned hardware profile is the reference environment for MoM V1
   formal runs?
2. How are approved exceptions to blocking metrics recorded in the publication
   classification?
3. Should the pack registry bind evaluators via Python entry points or a documented
   module-path convention until the runner slice lands?

## References

- [Epic #3038](https://github.com/vllm-project/semantic-router/issues/3038)
- [Proposal PR #3086](https://github.com/vllm-project/semantic-router/pull/3086)
- [MOM-01 child issue #3238](https://github.com/vllm-project/semantic-router/issues/3238)
- [Paused implementation PR #3083](https://github.com/vllm-project/semantic-router/pull/3083) (reuse after agreement and assignment)
- Existing EvalScope integration: `bench/router_flow/real_eval/`
