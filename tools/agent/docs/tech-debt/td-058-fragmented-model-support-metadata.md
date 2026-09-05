# TD058: Model Support Metadata Is Fragmented Across Runtime and Product Surfaces

## Status

Open.

## Owner Plan

[PL-0042: Unified Model Catalog and Evaluation Index](../plans/pl-0042-unified-model-catalog.md)

## Release Relevance

Not tied to a current release. The gap raises the cost and consistency risk of
every provider or model Day-0 change and blocks a trustworthy generated support
matrix and leaderboard.

## Scope

- Router provider profiles, protocols, model cards, reasoning behavior, and
  selection quality metadata;
- CLI built-in virtual-model catalog and physical-model references;
- Dashboard provider presets, model forms, and logo metadata;
- website provider/model support and evaluation presentation;
- training, Evaluation, Looper, and selection quality outputs.

## Summary

The repository has no single typed source for provider, protocol, model,
provider mapping, reasoning, presentation, benchmark, and score metadata. Provider
defaults are hard-coded in Router config helpers, the Dashboard owns a broader
independent provider catalog, the CLI catalog covers packaged virtual models,
and physical models are mostly free-form configuration or recommended strings.
Several runtime paths also assign incompatible meanings and fallbacks to one
scalar `quality_score`.

## Evidence

- [`src/semantic-router/pkg/config/helper.go`](../../../../src/semantic-router/pkg/config/helper.go)
  contains the runtime provider-type auth/path table.
- [`src/semantic-router/pkg/config/canonical_config.go`](../../../../src/semantic-router/pkg/config/canonical_config.go)
  owns logical model-card fields including `quality_score`.
- [`src/semantic-router/pkg/config/canonical_providers.go`](../../../../src/semantic-router/pkg/config/canonical_providers.go)
  separately owns backend bindings, pricing, API format, and reasoning-family
  references.
- [`dashboard/frontend/src/pages/modelProviderCatalog.ts`](../../../../dashboard/frontend/src/pages/modelProviderCatalog.ts)
  contains 40 provider/runtime UX presets independent of the Router registry.
- [`src/vllm-sr/cli/model_assets/latest/catalog.yaml`](../../../../src/vllm-sr/cli/model_assets/latest/catalog.yaml)
  contains five built-in virtual models and recommended physical-model strings,
  but no general provider/provider mapping/model-card graph.
- Selection, Looper, session-aware selection, training conversion, and
  Evaluation report code read, infer, or emit `quality_score` through different
  evidence rules.

## Why It Matters

- A Dashboard option can appear more strongly supported than its runtime
  contract warrants.
- New model support requires parallel edits and can omit reasoning, API,
  pricing, capability, logo, docs, or test metadata.
- Provider/model facts are mixed with operator bindings, making custom
  self-hosted models harder to represent cleanly.
- A scalar score cannot disclose benchmark versions, coverage, uncertainty, or
  source, and missing results can be mistaken for low quality.
- Central provider switches and broad config helpers grow with every Day-0
  addition.

## Desired End State

One versioned repository catalog defines protocols, providers, model cards,
provider mappings, reasoning behavior, presentation metadata, benchmarks, and composite
indices. A compiler merges optional user cards and evaluations into one
immutable effective registry with field provenance. Router, CLI, Dashboard, and
website artifacts are generated or materialized from that registry. Runtime
observations remain separate from static evidence, and the public Models page
accurately distinguishes native, compatible, runtime, virtual, and evaluated
support.

## Exit Criteria

- One schema-validated source graph feeds all runtime and presentation views.
- User model aliases explicitly reference canonical card identities, and
  built-in/custom cards share one materialization path.
- Built-in reasoning and protocol behavior no longer requires repeated user
  configuration.
- Provider auth/path defaults and Dashboard presets are generated from catalog
  definitions; semantic adapters are narrow and fixture-covered.
- Scalar `quality_score` and parameter-size quality fallbacks are replaced by
  versioned evaluation records, index results, coverage, uncertainty, and
  explicit missing-data policy.
- Dashboard Add Model and the website support matrix/leaderboards consume one
  sanitized generated snapshot while preserving provider logos.
- A representative Day-0 model change updates no independent runtime, UI, or
  documentation inventory.
