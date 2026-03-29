# Algorithm Tier Classification Design

**Issue:** [#1514 — Productionize model-selection algorithms without external-service-only dependencies](https://github.com/vllm-project/semantic-router/issues/1514)
**Date:** 2026-03-29
**Scope:** Classification and guardrails only. No algorithm functionality changes.

## Problem

Operators configuring model-selection algorithms have no way to distinguish production-ready algorithms from research experiments. All 12 algorithms appear equally available in config, but some depend on external services that aren't documented or packaged. A misconfigured algorithm silently degrades or fails at request time with no startup warning.

## Approach

Interface-driven tier classification. Each algorithm self-declares its tier and external dependencies via two new methods on the `Selector` interface. The factory uses these declarations at startup to emit warnings, check dependency health, and enrich metrics. A read-only `tier` field is added to the config schema for operator visibility.

## Design

### 1. Tier Model and Interface Changes

**Two tiers:**

- `supported` — production-ready, no surprise dependencies, well-tested
- `experimental` — research-grade, may require external services, limited production validation

**New types in `selector.go`:**

```go
type AlgorithmTier string

const (
    TierSupported    AlgorithmTier = "supported"
    TierExperimental AlgorithmTier = "experimental"
)

type Dependency struct {
    Name        string         // e.g., "AutoMix Verifier Server"
    Type        DependencyType // ExternalService, PretrainedModel, EmbeddingFunction
    Description string         // Human-readable purpose
    HealthURL   string         // Optional: URL to check at startup
    Required    bool           // true = degraded without it, false = optional enhancement
}

type DependencyType string

const (
    DependencyExternalService DependencyType = "external_service"
    DependencyPretrainedModel DependencyType = "pretrained_model"
    DependencyEmbeddingFunc   DependencyType = "embedding_function"
)
```

**Extended `Selector` interface:**

```go
type Selector interface {
    Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error)
    Method() SelectionMethod
    UpdateFeedback(ctx context.Context, feedback *Feedback) error
    Tier() AlgorithmTier              // NEW
    ExternalDependencies() []Dependency // NEW
}
```

**Tier assignments:**

| Algorithm     | Tier         | Rationale                                                    |
|---------------|--------------|--------------------------------------------------------------|
| static        | supported    | No deps, trivial                                             |
| elo           | supported    | No deps, well-tested                                         |
| router_dc     | supported    | Embedding only, well-understood                              |
| latency_aware | supported    | No deps, simple math                                         |
| hybrid        | supported    | Composite of supported algorithms                            |
| automix       | experimental | Optional external verifier server                            |
| rl_driven     | experimental | Optional external Router-R1 server                          |
| gmtrouter     | experimental | Requires pre-trained graph model, thin production test coverage |
| knn           | experimental | Requires pre-trained models, indirect test coverage only     |
| kmeans        | experimental | Requires pre-trained models, indirect test coverage only     |
| svm           | experimental | Requires pre-trained models, indirect test coverage only     |
| mlp           | experimental | Requires pre-trained models + GPU, indirect test coverage only |

### 2. Startup Behavior

When the factory creates selectors via `CreateAll()`, it will:

**Log tier and dependencies for every registered algorithm:**

```
[Selection] Registered algorithm: elo (tier=supported, dependencies=none)
[Selection] Registered algorithm: automix (tier=experimental, dependencies=[AutoMix Verifier Server (external_service)])
```

**Emit prominent warnings for experimental algorithms actually configured in a decision:**

```
[Selection] WARNING: Decision "my-decision" uses algorithm "automix" which is EXPERIMENTAL.
  This algorithm is not recommended for production use.
  External dependency: AutoMix Verifier Server (http://localhost:8090) — required for self-verification
```

Only warn for algorithms referenced by a decision in the config, not for all 12 registered in the global registry.

**Health-check external dependencies for configured experimental algorithms:**

For each configured experimental algorithm that declares a `HealthURL`, attempt a health check at startup (5-second timeout, reusing existing health check pattern). Log result clearly:

```
[Selection] Dependency check: AutoMix Verifier Server at http://localhost:8090 — UNREACHABLE (will degrade at runtime)
```

No startup failure in any case. Loud warnings only.

### 3. Config Schema Changes

**Read-only `tier` field** in per-algorithm config (informational, not operator-settable):

```yaml
decisions:
  - name: my-decision
    algorithm:
      type: automix
      tier: experimental   # read-only, injected by system
      automix:
        verification_threshold: 0.7
```

If an operator manually sets `tier:` in their config, it is silently ignored. The system always uses the code-declared tier from `Tier()`.

**Structured algorithm catalog in `routing_surface_catalog.go`:**

Replace the flat `supportedDecisionAlgorithmTypes` list:

```go
type AlgorithmCatalogEntry struct {
    Method SelectionMethod
    Tier   AlgorithmTier
}

var decisionAlgorithmCatalog = []AlgorithmCatalogEntry{
    {Method: "static", Tier: TierSupported},
    {Method: "elo", Tier: TierSupported},
    {Method: "router_dc", Tier: TierSupported},
    {Method: "latency_aware", Tier: TierSupported},
    {Method: "hybrid", Tier: TierSupported},
    {Method: "automix", Tier: TierExperimental},
    {Method: "rl_driven", Tier: TierExperimental},
    {Method: "gmtrouter", Tier: TierExperimental},
    {Method: "knn", Tier: TierExperimental},
    {Method: "kmeans", Tier: TierExperimental},
    {Method: "svm", Tier: TierExperimental},
    {Method: "mlp", Tier: TierExperimental},
}
```

Existing validation continues to accept all 12 types but can now emit a validation warning for experimental algorithms.

### 4. Metrics and Observability

**Add `tier` label to existing Prometheus metrics:**

- `model_selection_total` (counter)
- `model_selection_duration_seconds` (histogram)
- `model_selection_confidence` (gauge)

Example:

```
model_selection_total{method="automix",tier="experimental",decision="my-decision"} 42
model_selection_total{method="elo",tier="supported",decision="other-decision"} 1337
```

**New metric:**

- `model_selection_dependency_health` (gauge, 0 or 1) — emitted at startup per dependency per algorithm

```
model_selection_dependency_health{method="automix",dependency="AutoMix Verifier Server",type="external_service"} 0
```

**`SelectionResult` enrichment:**

Add `Tier AlgorithmTier` to `SelectionResult` so downstream code (extproc, logging) can include tier without re-lookup.

### 5. Test Strategy

No new test files. Changes integrate into existing test structure.

1. **Interface compliance** (`selector_test.go`): Table-driven test iterating registry from `CreateAll()`. Assert all 12 algorithms return valid `Tier()` and `ExternalDependencies()`. Assert supported algorithms return zero dependencies with `Required: true` and `Type: DependencyExternalService`.

2. **Startup warnings** (`selector_test.go`): Configure a decision with an experimental algorithm, capture log output, assert warning contains algorithm name, tier, and dependency info.

3. **Health check behavior** (`selector_test.go`): Mock unreachable health URL, verify startup completes without error and logs "UNREACHABLE" message.

4. **Metrics labels** (`metrics_test.go`): Verify `tier` label is present on selection metrics when an algorithm is invoked.

5. **Config catalog** (`validator_decision_test.go`): Verify structured `decisionAlgorithmCatalog` accepts all 12 types and maps to correct tiers.

**Out of scope:** No behavioral changes to any algorithm's `Select()` logic. No decision engine changes. No new E2E tests.

## Files Changed

| File | Change |
|------|--------|
| `pkg/selection/selector.go` | Add `AlgorithmTier`, `Dependency`, `DependencyType` types. Extend `Selector` interface. Add `Tier` to `SelectionResult`. |
| `pkg/selection/factory.go` | Add startup logging, tier warnings for configured decisions, dependency health checks. |
| `pkg/selection/static.go` | Implement `Tier()`, `ExternalDependencies()` |
| `pkg/selection/elo.go` | Implement `Tier()`, `ExternalDependencies()` |
| `pkg/selection/router_dc.go` | Implement `Tier()`, `ExternalDependencies()` |
| `pkg/selection/latency_aware.go` | Implement `Tier()`, `ExternalDependencies()` |
| `pkg/selection/hybrid.go` | Implement `Tier()`, `ExternalDependencies()` |
| `pkg/selection/automix.go` | Implement `Tier()`, `ExternalDependencies()` |
| `pkg/selection/rl_driven.go` | Implement `Tier()`, `ExternalDependencies()` |
| `pkg/selection/gmtrouter.go` | Implement `Tier()`, `ExternalDependencies()` |
| `pkg/selection/ml_adapter.go` | Implement `Tier()`, `ExternalDependencies()` for KNN/KMeans/SVM/MLP |
| `pkg/selection/metrics.go` | Add `tier` label to existing metrics. Add `model_selection_dependency_health` gauge. |
| `pkg/config/selection_config.go` | Add read-only `Tier` field to algorithm config structs. |
| `pkg/config/routing_surface_catalog.go` | Replace flat list with structured `decisionAlgorithmCatalog`. |
| `pkg/selection/selector_test.go` | Add interface compliance, startup warning, health check tests. |
| `pkg/selection/metrics_test.go` | Add tier label assertion. |
| `pkg/config/validator_decision_test.go` | Add catalog tier mapping test. |
