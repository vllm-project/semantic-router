package selection

import (
	"context"
	"strings"
	"testing"
)

func TestAlgorithmTier_Constants(t *testing.T) {
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
			adapter := NewMLSelectorAdapter(nil, method)

			if tier := adapter.Tier(); tier != TierExperimental {
				t.Errorf("%s.Tier() = %q, want %q", method, tier, TierExperimental)
			}

			deps := adapter.ExternalDependencies()
			if deps == nil {
				t.Errorf("%s.ExternalDependencies() returned nil, want non-nil", method)
			}

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

func TestSelect_PopulatesTier(t *testing.T) {
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
