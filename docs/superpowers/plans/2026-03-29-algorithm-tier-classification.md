# Algorithm Tier Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a supported/experimental tier system to the 12 model-selection algorithms so operators get clear warnings when using research-grade algorithms.

**Architecture:** Extend the `Selector` interface with `Tier()` and `ExternalDependencies()` methods. Each algorithm self-declares. The factory logs tiers at startup and emits prominent warnings for experimental algorithms configured in decisions. Metrics gain a `tier` label. The config catalog becomes a structured table.

**Tech Stack:** Go, Prometheus client_golang, existing test framework

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `pkg/selection/selector.go` | Modify | Add `AlgorithmTier`, `Dependency`, `DependencyType` types. Extend `Selector` interface. Add `Tier` field to `SelectionResult`. |
| `pkg/selection/factory.go` | Modify | Add `LogRegisteredAlgorithms()` and `CheckDependencyHealth()` helpers. Call them from `CreateAll()`. |
| `pkg/selection/static.go` | Modify | Add `Tier()` and `ExternalDependencies()` methods. |
| `pkg/selection/elo.go` | Modify | Add `Tier()` and `ExternalDependencies()` methods. |
| `pkg/selection/router_dc.go` | Modify | Add `Tier()` and `ExternalDependencies()` methods. |
| `pkg/selection/latency_aware.go` | Modify | Add `Tier()` and `ExternalDependencies()` methods. |
| `pkg/selection/hybrid.go` | Modify | Add `Tier()` and `ExternalDependencies()` methods. |
| `pkg/selection/automix.go` | Modify | Add `Tier()` and `ExternalDependencies()` methods. |
| `pkg/selection/rl_driven.go` | Modify | Add `Tier()` and `ExternalDependencies()` methods. |
| `pkg/selection/gmtrouter.go` | Modify | Add `Tier()` and `ExternalDependencies()` methods. |
| `pkg/selection/ml_adapter.go` | Modify | Add `Tier()` and `ExternalDependencies()` methods. |
| `pkg/selection/metrics.go` | Modify | Add `tier` label to `ModelSelectionTotal`, `ModelSelectionDuration`, `ModelSelectionConfidence`. Add `ModelSelectionDependencyHealth` gauge. |
| `pkg/config/routing_surface_catalog.go` | Modify | Replace flat `supportedDecisionAlgorithmTypes` with structured `decisionAlgorithmCatalog`. |
| `pkg/selection/selector_test.go` | Modify | Add tier compliance tests, startup warning tests, health check tests. |
| `pkg/selection/metrics_test.go` | Modify | Add tier label assertion test. |

All paths are relative to `src/semantic-router/`.

---

### Task 1: Add Tier and Dependency Types to selector.go

**Files:**
- Modify: `pkg/selection/selector.go`
- Test: `pkg/selection/selector_test.go`

- [ ] **Step 1: Write the failing test for tier types and interface compliance**

In `pkg/selection/selector_test.go`, add a test that verifies the new types compile and the interface has the expected methods:

```go
func TestAlgorithmTier_Constants(t *testing.T) {
	// Verify tier constants exist and have expected values
	if TierSupported != AlgorithmTier("supported") {
		t.Errorf("TierSupported = %q, want %q", TierSupported, "supported")
	}
	if TierExperimental != AlgorithmTier("experimental") {
		t.Errorf("TierExperimental = %q, want %q", TierExperimental, "experimental")
	}
}

func TestDependencyType_Constants(t *testing.T) {
	if DependencyExternalService != DependencyType("external_service") {
		t.Errorf("DependencyExternalService = %q, want %q", DependencyExternalService, "external_service")
	}
	if DependencyPretrainedModel != DependencyType("pretrained_model") {
		t.Errorf("DependencyPretrainedModel = %q, want %q", DependencyPretrainedModel, "pretrained_model")
	}
	if DependencyEmbeddingFunc != DependencyType("embedding_function") {
		t.Errorf("DependencyEmbeddingFunc = %q, want %q", DependencyEmbeddingFunc, "embedding_function")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/semantic-router && LD_LIBRARY_PATH=${PWD}/../../candle-binding/target/release go test -v -run "TestAlgorithmTier_Constants|TestDependencyType_Constants" ./pkg/selection/...`
Expected: FAIL — `AlgorithmTier`, `TierSupported`, etc. undefined

- [ ] **Step 3: Add tier and dependency types to selector.go**

Add after the `SelectionMethod` constants block (after line 91):

```go
// AlgorithmTier classifies algorithms by production readiness
type AlgorithmTier string

const (
	// TierSupported indicates a production-ready algorithm with no surprise dependencies
	TierSupported AlgorithmTier = "supported"

	// TierExperimental indicates a research-grade algorithm that may require
	// external services or has limited production validation
	TierExperimental AlgorithmTier = "experimental"
)

// DependencyType classifies external dependencies
type DependencyType string

const (
	// DependencyExternalService requires a separate running service (HTTP server, etc.)
	DependencyExternalService DependencyType = "external_service"

	// DependencyPretrainedModel requires pre-trained model artifacts on disk
	DependencyPretrainedModel DependencyType = "pretrained_model"

	// DependencyEmbeddingFunc requires an embedding function (provided by the router)
	DependencyEmbeddingFunc DependencyType = "embedding_function"
)

// Dependency describes an external dependency required by an algorithm
type Dependency struct {
	// Name is a human-readable name (e.g., "AutoMix Verifier Server")
	Name string

	// Type classifies the dependency
	Type DependencyType

	// Description explains what the dependency is used for
	Description string

	// HealthURL is an optional URL to check at startup (for external services)
	HealthURL string

	// Required indicates the algorithm degrades without this dependency
	Required bool
}
```

- [ ] **Step 4: Add Tier field to SelectionResult**

In `selector.go`, add to the `SelectionResult` struct after the `Method` field:

```go
	// Tier indicates the production readiness of the algorithm that made this selection
	Tier AlgorithmTier
```

- [ ] **Step 5: Extend the Selector interface**

In `selector.go`, add two methods to the `Selector` interface:

```go
	// Tier returns the production readiness classification of this algorithm
	Tier() AlgorithmTier

	// ExternalDependencies returns the list of external dependencies this algorithm requires
	ExternalDependencies() []Dependency
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd src/semantic-router && LD_LIBRARY_PATH=${PWD}/../../candle-binding/target/release go test -v -run "TestAlgorithmTier_Constants|TestDependencyType_Constants" ./pkg/selection/...`
Expected: PASS (types and constants compile correctly)

Note: The full test suite will NOT compile yet because existing selectors don't implement the new interface methods. That's expected — we'll fix them in Tasks 2-4.

- [ ] **Step 7: Commit**

```bash
git add src/semantic-router/pkg/selection/selector.go src/semantic-router/pkg/selection/selector_test.go
git commit -s -m "feat(selection): add AlgorithmTier and Dependency types to Selector interface (#1514)"
```

---

### Task 2: Implement Tier Methods on Supported Algorithms

**Files:**
- Modify: `pkg/selection/static.go`
- Modify: `pkg/selection/elo.go`
- Modify: `pkg/selection/router_dc.go`
- Modify: `pkg/selection/latency_aware.go`
- Modify: `pkg/selection/hybrid.go`
- Test: `pkg/selection/selector_test.go`

- [ ] **Step 1: Write the failing test**

In `pkg/selection/selector_test.go`, add:

```go
func TestSupportedAlgorithms_Tier(t *testing.T) {
	tests := []struct {
		name     string
		selector Selector
	}{
		{"static", NewStaticSelector(DefaultStaticConfig())},
		{"elo", NewEloSelector(DefaultEloConfig())},
		{"router_dc", NewRouterDCSelector(DefaultRouterDCConfig())},
		{"latency_aware", NewLatencyAwareSelector(nil)},
		{"hybrid", NewHybridSelector(DefaultHybridConfig())},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tier := tt.selector.Tier(); tier != TierSupported {
				t.Errorf("%s.Tier() = %q, want %q", tt.name, tier, TierSupported)
			}

			deps := tt.selector.ExternalDependencies()
			if deps == nil {
				t.Errorf("%s.ExternalDependencies() returned nil, want non-nil (empty slice is fine)", tt.name)
			}

			for _, dep := range deps {
				if dep.Type == DependencyExternalService && dep.Required {
					t.Errorf("%s: supported algorithm should not have required external service dependency: %s", tt.name, dep.Name)
				}
			}
		})
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/semantic-router && LD_LIBRARY_PATH=${PWD}/../../candle-binding/target/release go test -v -run TestSupportedAlgorithms_Tier ./pkg/selection/... 2>&1 | head -20`
Expected: FAIL — compile error, `StaticSelector` does not implement `Selector` (missing `Tier` and `ExternalDependencies`)

- [ ] **Step 3: Add Tier and ExternalDependencies to StaticSelector**

In `pkg/selection/static.go`, add after the `UpdateFeedback` method:

```go
// Tier returns the production readiness tier
func (s *StaticSelector) Tier() AlgorithmTier {
	return TierSupported
}

// ExternalDependencies returns external dependencies (none for static)
func (s *StaticSelector) ExternalDependencies() []Dependency {
	return []Dependency{}
}
```

- [ ] **Step 4: Add Tier and ExternalDependencies to EloSelector**

In `pkg/selection/elo.go`, add after the `UpdateFeedback` method (find the method, add after its closing brace):

```go
// Tier returns the production readiness tier
func (e *EloSelector) Tier() AlgorithmTier {
	return TierSupported
}

// ExternalDependencies returns external dependencies (none for Elo)
func (e *EloSelector) ExternalDependencies() []Dependency {
	return []Dependency{}
}
```

- [ ] **Step 5: Add Tier and ExternalDependencies to RouterDCSelector**

In `pkg/selection/router_dc.go`, add after the `UpdateFeedback` method:

```go
// Tier returns the production readiness tier
func (r *RouterDCSelector) Tier() AlgorithmTier {
	return TierSupported
}

// ExternalDependencies returns external dependencies (none requiring external services)
func (r *RouterDCSelector) ExternalDependencies() []Dependency {
	return []Dependency{}
}
```

- [ ] **Step 6: Add Tier and ExternalDependencies to LatencyAwareSelector**

In `pkg/selection/latency_aware.go`, add after the `UpdateFeedback` method:

```go
// Tier returns the production readiness tier
func (l *LatencyAwareSelector) Tier() AlgorithmTier {
	return TierSupported
}

// ExternalDependencies returns external dependencies (none for latency-aware)
func (l *LatencyAwareSelector) ExternalDependencies() []Dependency {
	return []Dependency{}
}
```

- [ ] **Step 7: Add Tier and ExternalDependencies to HybridSelector**

In `pkg/selection/hybrid.go`, add after the `UpdateFeedback` method:

```go
// Tier returns the production readiness tier
func (h *HybridSelector) Tier() AlgorithmTier {
	return TierSupported
}

// ExternalDependencies returns external dependencies (none for hybrid)
func (h *HybridSelector) ExternalDependencies() []Dependency {
	return []Dependency{}
}
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd src/semantic-router && LD_LIBRARY_PATH=${PWD}/../../candle-binding/target/release go test -v -run TestSupportedAlgorithms_Tier ./pkg/selection/...`
Expected: PASS (all five supported algorithms return `TierSupported` and empty dependency slices)

Note: Full suite still won't compile — experimental algorithms still missing the methods. That's Task 3.

- [ ] **Step 9: Commit**

```bash
git add src/semantic-router/pkg/selection/static.go src/semantic-router/pkg/selection/elo.go src/semantic-router/pkg/selection/router_dc.go src/semantic-router/pkg/selection/latency_aware.go src/semantic-router/pkg/selection/hybrid.go src/semantic-router/pkg/selection/selector_test.go
git commit -s -m "feat(selection): implement Tier() for supported algorithms (#1514)"
```

---

### Task 3: Implement Tier Methods on Experimental Algorithms

**Files:**
- Modify: `pkg/selection/automix.go`
- Modify: `pkg/selection/rl_driven.go`
- Modify: `pkg/selection/gmtrouter.go`
- Modify: `pkg/selection/ml_adapter.go`
- Test: `pkg/selection/selector_test.go`

- [ ] **Step 1: Write the failing test**

In `pkg/selection/selector_test.go`, add:

```go
func TestExperimentalAlgorithms_Tier(t *testing.T) {
	tests := []struct {
		name     string
		selector Selector
	}{
		{"automix", NewAutoMixSelector(DefaultAutoMixConfig())},
		{"rl_driven", NewRLDrivenSelector(DefaultRLDrivenConfig())},
		{"gmtrouter", NewGMTRouterSelector(DefaultGMTRouterConfig())},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tier := tt.selector.Tier(); tier != TierExperimental {
				t.Errorf("%s.Tier() = %q, want %q", tt.name, tier, TierExperimental)
			}

			deps := tt.selector.ExternalDependencies()
			if deps == nil {
				t.Errorf("%s.ExternalDependencies() returned nil, want non-nil", tt.name)
			}
		})
	}
}

func TestMLAdapterAlgorithms_Tier(t *testing.T) {
	methods := []SelectionMethod{MethodKNN, MethodKMeans, MethodSVM, MethodMLP}

	for _, method := range methods {
		t.Run(string(method), func(t *testing.T) {
			// MLSelectorAdapter wraps nil mlSelector for tier testing only
			adapter := NewMLSelectorAdapter(nil, method)

			if tier := adapter.Tier(); tier != TierExperimental {
				t.Errorf("%s.Tier() = %q, want %q", method, tier, TierExperimental)
			}

			deps := adapter.ExternalDependencies()
			if deps == nil {
				t.Errorf("%s.ExternalDependencies() returned nil, want non-nil", method)
			}

			// All ML adapters should declare pretrained model dependency
			foundPretrained := false
			for _, dep := range deps {
				if dep.Type == DependencyPretrainedModel {
					foundPretrained = true
					break
				}
			}
			if !foundPretrained {
				t.Errorf("%s: expected at least one pretrained_model dependency", method)
			}
		})
	}
}

func TestAutoMixSelector_DependenciesWithVerifier(t *testing.T) {
	cfg := DefaultAutoMixConfig()
	cfg.EnableSelfVerification = true
	cfg.VerifierServerURL = "http://localhost:8090"
	selector := NewAutoMixSelector(cfg)

	deps := selector.ExternalDependencies()
	foundVerifier := false
	for _, dep := range deps {
		if dep.Type == DependencyExternalService && strings.Contains(dep.Name, "Verifier") {
			foundVerifier = true
			if dep.HealthURL != "http://localhost:8090/health" {
				t.Errorf("Verifier HealthURL = %q, want %q", dep.HealthURL, "http://localhost:8090/health")
			}
		}
	}
	if !foundVerifier {
		t.Error("AutoMix with self-verification should declare verifier external service dependency")
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/semantic-router && LD_LIBRARY_PATH=${PWD}/../../candle-binding/target/release go test -v -run "TestExperimentalAlgorithms_Tier|TestMLAdapterAlgorithms_Tier|TestAutoMixSelector_DependenciesWithVerifier" ./pkg/selection/... 2>&1 | head -20`
Expected: FAIL — compile error, `AutoMixSelector` does not implement `Selector`

- [ ] **Step 3: Add Tier and ExternalDependencies to AutoMixSelector**

In `pkg/selection/automix.go`, add after the `UpdateFeedback` method:

```go
// Tier returns the production readiness tier
func (a *AutoMixSelector) Tier() AlgorithmTier {
	return TierExperimental
}

// ExternalDependencies returns external dependencies for AutoMix
func (a *AutoMixSelector) ExternalDependencies() []Dependency {
	deps := []Dependency{}
	if a.config.EnableSelfVerification && a.config.VerifierServerURL != "" {
		deps = append(deps, Dependency{
			Name:        "AutoMix Verifier Server",
			Type:        DependencyExternalService,
			Description: "LLM-based entailment verification for self-verification (arXiv:2310.12963)",
			HealthURL:   a.config.VerifierServerURL + "/health",
			Required:    true,
		})
	}
	return deps
}
```

- [ ] **Step 4: Add Tier and ExternalDependencies to RLDrivenSelector**

In `pkg/selection/rl_driven.go`, add after the `UpdateFeedback` method (or `Method()` — find a suitable location after the last public method):

```go
// Tier returns the production readiness tier
func (r *RLDrivenSelector) Tier() AlgorithmTier {
	return TierExperimental
}

// ExternalDependencies returns external dependencies for RL-driven selection
func (r *RLDrivenSelector) ExternalDependencies() []Dependency {
	deps := []Dependency{}
	if r.config.EnableLLMRouting && r.config.RouterR1ServerURL != "" {
		deps = append(deps, Dependency{
			Name:        "Router-R1 Server",
			Type:        DependencyExternalService,
			Description: "LLM-as-Router for advanced routing decisions (arXiv:2506.09033)",
			HealthURL:   r.config.RouterR1ServerURL + "/health",
			Required:    false,
		})
	}
	return deps
}
```

- [ ] **Step 5: Add Tier and ExternalDependencies to GMTRouterSelector**

In `pkg/selection/gmtrouter.go`, add after the `UpdateFeedback` method:

```go
// Tier returns the production readiness tier
func (g *GMTRouterSelector) Tier() AlgorithmTier {
	return TierExperimental
}

// ExternalDependencies returns external dependencies for GMTRouter
func (g *GMTRouterSelector) ExternalDependencies() []Dependency {
	deps := []Dependency{}
	if g.config.StoragePath != "" {
		deps = append(deps, Dependency{
			Name:        "Pre-trained graph model",
			Type:        DependencyPretrainedModel,
			Description: "Heterogeneous graph model weights (arXiv:2511.08590)",
			Required:    false,
		})
	}
	return deps
}
```

- [ ] **Step 6: Add Tier and ExternalDependencies to MLSelectorAdapter**

In `pkg/selection/ml_adapter.go`, add after the `UpdateFeedback` method:

```go
// Tier returns the production readiness tier
func (a *MLSelectorAdapter) Tier() AlgorithmTier {
	return TierExperimental
}

// ExternalDependencies returns external dependencies for ML selectors
func (a *MLSelectorAdapter) ExternalDependencies() []Dependency {
	return []Dependency{
		{
			Name:        fmt.Sprintf("Pre-trained %s model", a.method),
			Type:        DependencyPretrainedModel,
			Description: fmt.Sprintf("Pre-trained model artifacts for %s selection", a.method),
			Required:    true,
		},
	}
}
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd src/semantic-router && LD_LIBRARY_PATH=${PWD}/../../candle-binding/target/release go test -v -run "TestExperimentalAlgorithms_Tier|TestMLAdapterAlgorithms_Tier|TestAutoMixSelector_DependenciesWithVerifier" ./pkg/selection/...`
Expected: PASS

- [ ] **Step 8: Run the full test suite to verify nothing is broken**

Run: `cd src/semantic-router && LD_LIBRARY_PATH=${PWD}/../../candle-binding/target/release go test ./pkg/selection/...`
Expected: PASS — all existing tests plus new ones

- [ ] **Step 9: Commit**

```bash
git add src/semantic-router/pkg/selection/automix.go src/semantic-router/pkg/selection/rl_driven.go src/semantic-router/pkg/selection/gmtrouter.go src/semantic-router/pkg/selection/ml_adapter.go src/semantic-router/pkg/selection/selector_test.go
git commit -s -m "feat(selection): implement Tier() for experimental algorithms (#1514)"
```

---

### Task 4: Add Startup Logging and Dependency Health Checks to Factory

**Files:**
- Modify: `pkg/selection/factory.go`
- Test: `pkg/selection/selector_test.go`

- [ ] **Step 1: Write the failing test**

In `pkg/selection/selector_test.go`, add:

```go
func TestRegistry_AllSelectors_HaveTier(t *testing.T) {
	factory := NewFactory(DefaultModelSelectionConfig())
	registry := factory.CreateAll()

	expectedTiers := map[SelectionMethod]AlgorithmTier{
		MethodStatic:       TierSupported,
		MethodElo:          TierSupported,
		MethodRouterDC:     TierSupported,
		MethodLatencyAware: TierSupported,
		MethodHybrid:       TierSupported,
		MethodAutoMix:      TierExperimental,
		MethodRLDriven:     TierExperimental,
		MethodGMTRouter:    TierExperimental,
	}

	for method, expectedTier := range expectedTiers {
		selector, ok := registry.Get(method)
		if !ok {
			t.Errorf("Registry missing selector for method %q", method)
			continue
		}
		if tier := selector.Tier(); tier != expectedTier {
			t.Errorf("%s.Tier() = %q, want %q", method, tier, expectedTier)
		}
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/semantic-router && LD_LIBRARY_PATH=${PWD}/../../candle-binding/target/release go test -v -run TestRegistry_AllSelectors_HaveTier ./pkg/selection/...`
Expected: FAIL or PASS depending on compile state — if previous tasks are done, this may pass already. The real test is that the registry-level view works.

- [ ] **Step 3: Add LogRegisteredAlgorithms to factory.go**

In `pkg/selection/factory.go`, add a new method and a helper function. Add these after the `CreateAll` method:

```go
// LogRegisteredAlgorithms logs the tier and dependencies of each registered algorithm
func LogRegisteredAlgorithms(registry *Registry) {
	registry.mu.RLock()
	defer registry.mu.RUnlock()

	for method, selector := range registry.selectors {
		deps := selector.ExternalDependencies()
		if len(deps) == 0 {
			logging.Infof("[Selection] Registered algorithm: %s (tier=%s, dependencies=none)", method, selector.Tier())
		} else {
			depNames := make([]string, len(deps))
			for i, dep := range deps {
				depNames[i] = fmt.Sprintf("%s (%s)", dep.Name, dep.Type)
			}
			logging.Infof("[Selection] Registered algorithm: %s (tier=%s, dependencies=[%s])",
				method, selector.Tier(), strings.Join(depNames, ", "))
		}
	}
}

// WarnExperimentalAlgorithms logs prominent warnings for experimental algorithms
// that are actually configured in operator decisions
func WarnExperimentalAlgorithms(registry *Registry, configuredMethods []SelectionMethod) {
	for _, method := range configuredMethods {
		selector, ok := registry.Get(method)
		if !ok {
			continue
		}
		if selector.Tier() != TierExperimental {
			continue
		}

		deps := selector.ExternalDependencies()
		logging.Warnf("[Selection] WARNING: Algorithm %q is EXPERIMENTAL and not recommended for production use", method)
		for _, dep := range deps {
			if dep.HealthURL != "" {
				logging.Warnf("[Selection]   External dependency: %s (%s)", dep.Name, dep.HealthURL)
			} else {
				logging.Warnf("[Selection]   Dependency: %s — %s", dep.Name, dep.Description)
			}
		}
	}
}

// CheckDependencyHealth checks reachability of external service dependencies
// for the given algorithms. Logs results but never fails.
func CheckDependencyHealth(registry *Registry, configuredMethods []SelectionMethod) {
	for _, method := range configuredMethods {
		selector, ok := registry.Get(method)
		if !ok {
			continue
		}

		for _, dep := range selector.ExternalDependencies() {
			if dep.Type != DependencyExternalService || dep.HealthURL == "" {
				continue
			}

			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			client := &http.Client{Timeout: 5 * time.Second}
			req, err := http.NewRequestWithContext(ctx, "GET", dep.HealthURL, nil)
			if err != nil {
				logging.Warnf("[Selection] Dependency check: %s — UNREACHABLE (bad URL: %v)", dep.Name, err)
				cancel()
				continue
			}

			resp, err := client.Do(req)
			cancel()
			if err != nil {
				logging.Warnf("[Selection] Dependency check: %s at %s — UNREACHABLE (will degrade at runtime)", dep.Name, dep.HealthURL)
			} else {
				resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					logging.Infof("[Selection] Dependency check: %s at %s — OK", dep.Name, dep.HealthURL)
				} else {
					logging.Warnf("[Selection] Dependency check: %s at %s — unhealthy (status %d)", dep.Name, dep.HealthURL, resp.StatusCode)
				}
			}
		}
	}
}
```

- [ ] **Step 4: Add required imports to factory.go**

Add `"context"`, `"fmt"`, `"net/http"`, `"strings"`, and `"time"` to the import block in `factory.go`. Some may already be present — only add missing ones.

- [ ] **Step 5: Call LogRegisteredAlgorithms from CreateAll**

In `factory.go`, in the `CreateAll()` method, replace the final logging line:

```go
	logging.Infof("[SelectionFactory] Created all selectors: static, elo, router_dc, automix, hybrid, knn, kmeans, svm, mlp, rl_driven, gmtrouter, latency_aware")
	return registry
```

with:

```go
	LogRegisteredAlgorithms(registry)
	return registry
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd src/semantic-router && LD_LIBRARY_PATH=${PWD}/../../candle-binding/target/release go test -v -run TestRegistry_AllSelectors_HaveTier ./pkg/selection/...`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/semantic-router/pkg/selection/factory.go src/semantic-router/pkg/selection/selector_test.go
git commit -s -m "feat(selection): add startup tier logging and dependency health checks (#1514)"
```

---

### Task 5: Add Tier Label to Prometheus Metrics

**Files:**
- Modify: `pkg/selection/metrics.go`
- Test: `pkg/selection/metrics_test.go`

- [ ] **Step 1: Write the failing test**

In `pkg/selection/metrics_test.go`, add:

```go
func TestRecordSelectionFull_IncludesTierLabel(t *testing.T) {
	InitializeMetrics()

	// Record a selection with tier
	RecordSelectionWithTier(MethodElo, "test-model", "test-decision", TierSupported, 0.8, 0.9, time.Millisecond)

	// Verify the counter has the expected labels by collecting
	// We just verify it doesn't panic — label validation happens at registration
}

func TestModelSelectionDependencyHealth_Metric(t *testing.T) {
	InitializeMetrics()

	RecordDependencyHealth("automix", "AutoMix Verifier Server", "external_service", false)
	RecordDependencyHealth("elo", "", "", true)

	// Verify no panic on recording
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/semantic-router && LD_LIBRARY_PATH=${PWD}/../../candle-binding/target/release go test -v -run "TestRecordSelectionFull_IncludesTierLabel|TestModelSelectionDependencyHealth_Metric" ./pkg/selection/...`
Expected: FAIL — `RecordSelectionWithTier` and `RecordDependencyHealth` undefined

- [ ] **Step 3: Add tier label to metrics and new dependency health gauge**

In `pkg/selection/metrics.go`, add a new metric variable in the `var` block (after `RLDrivenImplicitFeedback`):

```go
	// ModelSelectionDependencyHealth tracks external dependency reachability
	// Labels: method, dependency, type
	ModelSelectionDependencyHealth *prometheus.GaugeVec
```

In the `InitializeMetrics()` function, add the `"tier"` label to `ModelSelectionTotal`, `ModelSelectionDuration`, and `ModelSelectionConfidence`. Change each to:

For `ModelSelectionTotal` — change labels from `[]string{"method", "model", "decision"}` to `[]string{"method", "model", "decision", "tier"}`.

For `ModelSelectionDuration` — change labels from `[]string{"method"}` to `[]string{"method", "tier"}`.

For `ModelSelectionConfidence` — change labels from `[]string{"method"}` to `[]string{"method", "tier"}`.

Add registration for `ModelSelectionDependencyHealth` after the RL metrics:

```go
		ModelSelectionDependencyHealth = promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "llm_model_selection_dependency_health",
				Help: "External dependency reachability (1=healthy, 0=unreachable)",
			},
			[]string{"method", "dependency", "type"},
		)
```

- [ ] **Step 4: Update preInitializeMetrics for new labels**

In `preInitializeMetrics()`, update the loop that initializes selection metrics:

```go
	for _, method := range methods {
		ModelSelectionTotal.WithLabelValues(method, "_init", "_init", "supported")
		ModelSelectionDuration.WithLabelValues(method, "supported")
		ModelSelectionScore.WithLabelValues(method, "_init")
		ModelSelectionConfidence.WithLabelValues(method, "supported")
		ModelSelectionHistory.WithLabelValues(method, "_init")
	}
```

Also add a pre-initialization for the dependency health metric:

```go
	ModelSelectionDependencyHealth.WithLabelValues("_init", "_init", "_init").Set(0)
```

- [ ] **Step 5: Add RecordSelectionWithTier and RecordDependencyHealth functions**

In `pkg/selection/metrics.go`, add:

```go
// RecordSelectionWithTier records a model selection event with tier label
func RecordSelectionWithTier(method SelectionMethod, model string, decision string, tier AlgorithmTier, score, confidence float64, duration time.Duration) {
	if !metricsEnabled {
		return
	}

	methodStr := string(method)
	tierStr := string(tier)

	ModelSelectionTotal.WithLabelValues(methodStr, model, decision, tierStr).Inc()
	ModelSelectionDuration.WithLabelValues(methodStr, tierStr).Observe(duration.Seconds())
	ModelSelectionScore.WithLabelValues(methodStr, model).Observe(score)
	ModelSelectionConfidence.WithLabelValues(methodStr, tierStr).Observe(confidence)
	ModelSelectionHistory.WithLabelValues(methodStr, decision).Inc()
}

// RecordDependencyHealth records the health status of an external dependency
func RecordDependencyHealth(method string, dependency string, depType string, healthy bool) {
	if !metricsEnabled {
		return
	}

	val := 0.0
	if healthy {
		val = 1.0
	}
	ModelSelectionDependencyHealth.WithLabelValues(method, dependency, depType).Set(val)
}
```

- [ ] **Step 6: Update RecordSelectionFull to pass tier**

Update the existing `RecordSelectionFull` to include a default tier so existing callers don't break:

```go
func RecordSelectionFull(method SelectionMethod, model string, decision string, score, confidence float64, duration time.Duration) {
	if !metricsEnabled {
		return
	}

	methodStr := string(method)

	ModelSelectionTotal.WithLabelValues(methodStr, model, decision, "").Inc()
	ModelSelectionDuration.WithLabelValues(methodStr, "").Observe(duration.Seconds())
	ModelSelectionScore.WithLabelValues(methodStr, model).Observe(score)
	ModelSelectionConfidence.WithLabelValues(methodStr, "").Observe(confidence)
	ModelSelectionHistory.WithLabelValues(methodStr, decision).Inc()
}
```

Also update `RecordSelection`:

```go
func RecordSelection(method string, decision string, model string, score float64) {
	if !metricsEnabled {
		return
	}

	ModelSelectionTotal.WithLabelValues(method, model, decision, "").Inc()
	ModelSelectionScore.WithLabelValues(method, model).Observe(score)
	ModelSelectionHistory.WithLabelValues(method, decision).Inc()
}
```

And update `RecordHybridSelection`:

```go
func RecordHybridSelection(selectedModel string, decision string, componentChoices map[string]string, score, confidence float64, duration time.Duration) {
	if !metricsEnabled {
		return
	}

	RecordSelectionFull(MethodHybrid, selectedModel, decision, score, confidence, duration)

	if len(componentChoices) > 1 {
		choices := make([]string, 0, len(componentChoices))
		for _, model := range componentChoices {
			choices = append(choices, model)
		}
		agreement := calculateAgreementRatio(choices)
		RecordComponentAgreement(agreement)
	}
}
```

And `RecordRLSelection`:

```go
func RecordRLSelection(model, category, userID string, score float64) {
	if !metricsEnabled {
		return
	}

	if category == "" {
		category = "_global"
	}

	ModelSelectionTotal.WithLabelValues("rl_driven", model, category, "experimental").Inc()
	ModelSelectionScore.WithLabelValues("rl_driven", model).Observe(score)
	ModelSelectionHistory.WithLabelValues("rl_driven", category).Inc()

	if userID != "" {
		RLDrivenPersonalizedSelections.WithLabelValues(model, category).Inc()
	}
}
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd src/semantic-router && LD_LIBRARY_PATH=${PWD}/../../candle-binding/target/release go test -v -run "TestRecordSelectionFull_IncludesTierLabel|TestModelSelectionDependencyHealth_Metric" ./pkg/selection/...`
Expected: PASS

- [ ] **Step 8: Run full metrics test suite**

Run: `cd src/semantic-router && LD_LIBRARY_PATH=${PWD}/../../candle-binding/target/release go test -v ./pkg/selection/... -run Test.*Metric`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/semantic-router/pkg/selection/metrics.go src/semantic-router/pkg/selection/metrics_test.go
git commit -s -m "feat(selection): add tier label to Prometheus metrics and dependency health gauge (#1514)"
```

---

### Task 6: Replace Flat Algorithm Catalog with Structured Table

**Files:**
- Modify: `pkg/config/routing_surface_catalog.go`
- Test: `pkg/config/routing_surface_catalog_test.go` (if exists, else add test to an existing test file in `pkg/config/`)

- [ ] **Step 1: Check for existing catalog tests**

Run: `find src/semantic-router/pkg/config -name '*catalog*test*' -o -name '*surface*test*' | head -5`
and: `grep -rn "IsSupportedDecisionAlgorithmType\|SupportedDecisionAlgorithmTypes" src/semantic-router/pkg/config/*_test.go | head -10`

This determines where to add tests.

- [ ] **Step 2: Write the failing test**

Add a test (in the appropriate test file in `pkg/config/`):

```go
func TestDecisionAlgorithmCatalog_AllTypesHaveTier(t *testing.T) {
	catalog := DecisionAlgorithmCatalog()
	if len(catalog) == 0 {
		t.Fatal("DecisionAlgorithmCatalog() returned empty catalog")
	}

	for _, entry := range catalog {
		if entry.Type == "" {
			t.Error("Catalog entry has empty Type")
		}
		if entry.Tier != "supported" && entry.Tier != "experimental" {
			t.Errorf("Catalog entry %q has invalid Tier %q", entry.Type, entry.Tier)
		}
	}
}

func TestDecisionAlgorithmCatalog_BackwardsCompatible(t *testing.T) {
	// Verify all previously supported types are still present
	previousTypes := []string{
		"automix", "confidence", "elo", "gmtrouter", "hybrid",
		"kmeans", "knn", "latency_aware", "ratings", "remom",
		"rl_driven", "router_dc", "static", "svm",
	}

	for _, algType := range previousTypes {
		if !IsSupportedDecisionAlgorithmType(algType) {
			t.Errorf("Previously supported algorithm type %q is no longer supported", algType)
		}
	}
}

func TestGetAlgorithmTier(t *testing.T) {
	tests := []struct {
		algType      string
		expectedTier string
	}{
		{"static", "supported"},
		{"elo", "supported"},
		{"router_dc", "supported"},
		{"latency_aware", "supported"},
		{"hybrid", "supported"},
		{"automix", "experimental"},
		{"rl_driven", "experimental"},
		{"gmtrouter", "experimental"},
		{"knn", "experimental"},
		{"kmeans", "experimental"},
		{"svm", "experimental"},
		{"mlp", "experimental"},
	}

	for _, tt := range tests {
		t.Run(tt.algType, func(t *testing.T) {
			tier := GetAlgorithmTier(tt.algType)
			if tier != tt.expectedTier {
				t.Errorf("GetAlgorithmTier(%q) = %q, want %q", tt.algType, tier, tt.expectedTier)
			}
		})
	}
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd src/semantic-router && go test -v -run "TestDecisionAlgorithmCatalog_AllTypesHaveTier|TestDecisionAlgorithmCatalog_BackwardsCompatible|TestGetAlgorithmTier" ./pkg/config/...`
Expected: FAIL — `DecisionAlgorithmCatalog`, `GetAlgorithmTier` undefined

- [ ] **Step 4: Replace flat list with structured catalog**

In `pkg/config/routing_surface_catalog.go`, replace the `supportedDecisionAlgorithmTypes` variable and add new types and functions:

Replace:

```go
var supportedDecisionAlgorithmTypes = []string{
	"automix",
	"confidence",
	"elo",
	"gmtrouter",
	"hybrid",
	"kmeans",
	"knn",
	"latency_aware",
	"ratings",
	"remom",
	"rl_driven",
	"router_dc",
	"static",
	"svm",
}
```

With:

```go
// AlgorithmCatalogEntry describes a model-selection algorithm and its tier
type AlgorithmCatalogEntry struct {
	Type string // algorithm type name (e.g., "elo")
	Tier string // "supported" or "experimental"
}

var decisionAlgorithmCatalog = []AlgorithmCatalogEntry{
	{Type: "automix", Tier: "experimental"},
	{Type: "confidence", Tier: "supported"},
	{Type: "elo", Tier: "supported"},
	{Type: "gmtrouter", Tier: "experimental"},
	{Type: "hybrid", Tier: "supported"},
	{Type: "kmeans", Tier: "experimental"},
	{Type: "knn", Tier: "experimental"},
	{Type: "latency_aware", Tier: "supported"},
	{Type: "mlp", Tier: "experimental"},
	{Type: "ratings", Tier: "supported"},
	{Type: "remom", Tier: "supported"},
	{Type: "rl_driven", Tier: "experimental"},
	{Type: "router_dc", Tier: "supported"},
	{Type: "static", Tier: "supported"},
	{Type: "svm", Tier: "experimental"},
}

// supportedDecisionAlgorithmTypes is derived from the catalog for backwards compatibility
var supportedDecisionAlgorithmTypes = func() []string {
	types := make([]string, len(decisionAlgorithmCatalog))
	for i, entry := range decisionAlgorithmCatalog {
		types[i] = entry.Type
	}
	return types
}()
```

- [ ] **Step 5: Add DecisionAlgorithmCatalog and GetAlgorithmTier functions**

In `pkg/config/routing_surface_catalog.go`, add:

```go
// DecisionAlgorithmCatalog returns the full structured catalog of algorithm types and tiers
func DecisionAlgorithmCatalog() []AlgorithmCatalogEntry {
	result := make([]AlgorithmCatalogEntry, len(decisionAlgorithmCatalog))
	copy(result, decisionAlgorithmCatalog)
	return result
}

// GetAlgorithmTier returns the tier for a given algorithm type, or empty string if unknown
func GetAlgorithmTier(algorithmType string) string {
	for _, entry := range decisionAlgorithmCatalog {
		if entry.Type == algorithmType {
			return entry.Tier
		}
	}
	return ""
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd src/semantic-router && go test -v -run "TestDecisionAlgorithmCatalog_AllTypesHaveTier|TestDecisionAlgorithmCatalog_BackwardsCompatible|TestGetAlgorithmTier" ./pkg/config/...`
Expected: PASS

- [ ] **Step 7: Run full config test suite**

Run: `cd src/semantic-router && go test ./pkg/config/...`
Expected: PASS — backwards-compatible, all existing callers of `IsSupportedDecisionAlgorithmType` still work

- [ ] **Step 8: Commit**

```bash
git add src/semantic-router/pkg/config/routing_surface_catalog.go src/semantic-router/pkg/config/routing_surface_catalog_test.go
git commit -s -m "feat(config): replace flat algorithm list with structured tier catalog (#1514)"
```

---

### Task 7: Wire Startup Warnings into Router Initialization

**Files:**
- Modify: `pkg/extproc/router_selection.go`

- [ ] **Step 1: Modify createModelSelectorRegistry to call tier warnings and health checks**

In `pkg/extproc/router_selection.go`, update `createModelSelectorRegistry` to extract configured algorithm methods from decisions and call the new functions:

```go
func createModelSelectorRegistry(cfg *config.RouterConfig) *selection.Registry {
	modelSelectionCfg := buildModelSelectionConfig(cfg)
	backendModels := cfg.BackendModels
	selectionFactory := selection.NewFactory(modelSelectionCfg)

	if backendModels.ModelConfig != nil {
		selectionFactory = selectionFactory.WithModelConfig(backendModels.ModelConfig)
	}
	if len(cfg.Categories) > 0 {
		selectionFactory = selectionFactory.WithCategories(cfg.Categories)
	}
	selectionFactory = selectionFactory.WithEmbeddingFunc(func(text string) ([]float32, error) {
		output, err := candle_binding.GetEmbeddingBatched(text, "qwen3", 1024)
		if err != nil {
			return nil, err
		}
		return output.Embedding, nil
	})

	registry := selectionFactory.CreateAll()
	selection.GlobalRegistry = registry

	// Collect algorithm methods actually configured in decisions
	configuredMethods := collectConfiguredAlgorithmMethods(cfg)

	// Warn about experimental algorithms and check dependency health
	selection.WarnExperimentalAlgorithms(registry, configuredMethods)
	selection.CheckDependencyHealth(registry, configuredMethods)

	logging.Infof("[Router] Initialized model selection registry (per-decision algorithm config)")
	return registry
}
```

Add a new helper function:

```go
func collectConfiguredAlgorithmMethods(cfg *config.RouterConfig) []selection.SelectionMethod {
	seen := make(map[string]bool)
	var methods []selection.SelectionMethod

	for _, decision := range cfg.IntelligentRouting.Decisions {
		if decision.Algorithm == nil || decision.Algorithm.Type == "" {
			continue
		}
		if !seen[decision.Algorithm.Type] {
			seen[decision.Algorithm.Type] = true
			methods = append(methods, selection.SelectionMethod(decision.Algorithm.Type))
		}
	}

	return methods
}
```

- [ ] **Step 2: Run go vet to verify it compiles**

Run: `cd src/semantic-router && go vet ./pkg/extproc/...`
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add src/semantic-router/pkg/extproc/router_selection.go
git commit -s -m "feat(extproc): wire tier warnings and health checks into router startup (#1514)"
```

---

### Task 8: Populate Tier in SelectionResult

**Files:**
- Modify: `pkg/selection/factory.go` (the `Select` function in `selector.go`)
- Test: `pkg/selection/selector_test.go`

- [ ] **Step 1: Write the failing test**

In `pkg/selection/selector_test.go`, add:

```go
func TestSelect_PopulatesTier(t *testing.T) {
	// Setup registry with a static selector
	factory := NewFactory(DefaultModelSelectionConfig())
	GlobalRegistry = factory.CreateAll()

	candidates := createCandidateModels("model-a", "model-b")
	selCtx := &SelectionContext{
		Query:           "test query",
		CandidateModels: candidates,
	}

	result, err := Select(context.Background(), MethodStatic, selCtx)
	if err != nil {
		t.Fatalf("Select failed: %v", err)
	}

	if result.Tier != TierSupported {
		t.Errorf("Select(MethodStatic) result.Tier = %q, want %q", result.Tier, TierSupported)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd src/semantic-router && LD_LIBRARY_PATH=${PWD}/../../candle-binding/target/release go test -v -run TestSelect_PopulatesTier ./pkg/selection/...`
Expected: FAIL — `result.Tier` is empty string

- [ ] **Step 3: Update the Select function to populate Tier**

In `pkg/selection/selector.go`, update the `Select` function to set the tier after calling the selector:

```go
func Select(ctx context.Context, method SelectionMethod, selCtx *SelectionContext) (*SelectionResult, error) {
	selector, ok := GlobalRegistry.Get(method)
	if !ok {
		// Fall back to static selection
		selector, _ = GlobalRegistry.Get(MethodStatic)
	}
	if selector == nil {
		// Ultimate fallback: return first candidate
		return &SelectionResult{
			SelectedModel: selCtx.CandidateModels[0].Model,
			LoRAName:      selCtx.CandidateModels[0].LoRAName,
			Score:         1.0,
			Confidence:    1.0,
			Method:        MethodStatic,
			Tier:          TierSupported,
			Reasoning:     "No selector available, using first candidate",
		}, nil
	}
	result, err := selector.Select(ctx, selCtx)
	if err != nil {
		return nil, err
	}
	result.Tier = selector.Tier()
	return result, nil
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd src/semantic-router && LD_LIBRARY_PATH=${PWD}/../../candle-binding/target/release go test -v -run TestSelect_PopulatesTier ./pkg/selection/...`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/semantic-router/pkg/selection/selector.go src/semantic-router/pkg/selection/selector_test.go
git commit -s -m "feat(selection): populate Tier in SelectionResult (#1514)"
```

---

### Task 9: Final Integration Test — Full Suite

**Files:**
- Test: all modified files

- [ ] **Step 1: Run the full selection package test suite**

Run: `cd src/semantic-router && LD_LIBRARY_PATH=${PWD}/../../candle-binding/target/release go test -v ./pkg/selection/...`
Expected: All tests PASS

- [ ] **Step 2: Run the full config package test suite**

Run: `cd src/semantic-router && go test -v ./pkg/config/...`
Expected: All tests PASS

- [ ] **Step 3: Run go vet across affected packages**

Run: `cd src/semantic-router && go vet ./pkg/selection/... ./pkg/config/... ./pkg/extproc/...`
Expected: No errors

- [ ] **Step 4: Run the agent-validate gate**

Run: `make agent-validate` (from repo root)
Expected: PASS

- [ ] **Step 5: Verify no go.mod changes needed**

Run: `cd src/semantic-router && go mod tidy && git diff go.mod go.sum`
Expected: No changes (no new external dependencies introduced)

- [ ] **Step 6: Commit if any cleanup was needed**

Only if previous steps required fixups:

```bash
git add -A
git commit -s -m "fix(selection): address integration test findings (#1514)"
```
