# TD044: Router-Owned Model Lifecycle Is Split Across Download, Runtime, and API Surfaces

## Status

Open - narrowed by the model lifecycle catalog consolidation.

## Owner Plan

PL0032 Architecture Debt Consolidation

## Release Relevance

None - non-release debt

## Scope

`src/semantic-router/pkg/modeldownload/**`,
`src/semantic-router/pkg/modelruntime/**`,
`src/semantic-router/pkg/modellifecycle/**`,
`src/semantic-router/pkg/apiserver/route_model_info*.go`,
`src/semantic-router/pkg/extproc/router_memory.go`, canonical config defaults,
and AMD-local model validation paths.

## Summary

Router-owned model assets are now bound through a config-owned lifecycle
catalog and materialized into a shared `pkg/modellifecycle` plan. Canonical
defaults, system-model bindings, companion mapping files, downloader specs,
embedding-runtime path resolution, memory embedding selection, and model-info
API placeholders consume that shared model plan.

The remaining debt is narrower: legacy model registry metadata still serves
alias lookup and Hugging Face repo IDs, and some non-router examples/tests still
mention MiniLM-era external embedding models. Those references must not become
canonical router defaults again.

## Evidence

- [src/semantic-router/pkg/modeldownload/config_parser.go](../../../src/semantic-router/pkg/modeldownload/config_parser.go)
- [src/semantic-router/pkg/config/model_lifecycle_catalog.go](../../../src/semantic-router/pkg/config/model_lifecycle_catalog.go)
- [src/semantic-router/pkg/modelruntime/router_runtime.go](../../../src/semantic-router/pkg/modelruntime/router_runtime.go)
- [src/semantic-router/pkg/modellifecycle/plan.go](../../../src/semantic-router/pkg/modellifecycle/plan.go)
- [src/semantic-router/pkg/apiserver/route_model_info_classifiers.go](../../../src/semantic-router/pkg/apiserver/route_model_info_classifiers.go)
- [src/semantic-router/pkg/apiserver/route_model_info_embeddings.go](../../../src/semantic-router/pkg/apiserver/route_model_info_embeddings.go)
- [src/semantic-router/pkg/extproc/router_memory.go](../../../src/semantic-router/pkg/extproc/router_memory.go)
- [deploy/recipes/balance.yaml](../../../deploy/recipes/balance.yaml)

## Why It Matters

- Registry metadata, default bindings, download planning, initialization, and
  API model reporting can drift and disagree about which router-owned models are
  actually required.
- Hardcoded model fallbacks bypass lifecycle-owned local assets, so operators
  cannot reason about whether an asset is pre-mounted, downloaded, initialized,
  or only referenced.
- AMD-local behavior depends on model lifecycle clarity: sparse global overrides
  should keep canonical model bindings while changing device policy, and smoke
  configs should be able to intentionally clear router-owned model downloads.

## Desired End State

- One shared lifecycle catalog and plan own router-owned model assets, default
  bindings, required companion files, feature gates, and embedding model
  selection.
- Downloader, runtime initialization, startup status, API model-info routes, and
  tests consume that plan instead of re-deriving inventories independently.
- BERT fallback behavior resolves through the registry-managed
  `models/mom-embedding-light` asset instead of a hardcoded Hugging Face model
  ID.
- AMD-local validation proves the canonical balance recipe downloads and
  initializes the expected router-owned models on ROCm defaults.

## Exit Criteria

- Canonical defaults, `modeldownload.BuildModelSpecs`,
  `modelruntime.PrepareRouterRuntime`, memory embedding selection, and
  model-info API assembly consume the shared lifecycle catalog/plan.
- No production path uses reflection to discover model paths from config.
- No production runtime code hardcodes `sentence-transformers/all-MiniLM-L6-v2`
  or `sentence-transformers/all-MiniLM-L12-v2`; repo IDs live in the registry.
- `go test` coverage proves lifecycle binding registry coverage, canonical
  defaults, BERT fallback, and AMD balance recipe model plans.
- `make agent-feature-gate ENV=cpu CHANGED_FILES="..."` passes.
- `make agent-feature-gate ENV=amd CHANGED_FILES="..."` or the documented
  AMD-local equivalent passes on an AMD/ROCm machine.
