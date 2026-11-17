package selection

import (
	"context"
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
}

func TestSupportedAlgorithms_Tier(t *testing.T) {
	tests := []struct {
		name     string
		selector Selector
	}{
		{"static", NewStaticSelector(DefaultStaticConfig())},
		{"multi_factor", NewMultiFactorSelector(nil)},
		{"random", NewRandomSelector()},
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

func TestRegistry_AllSelectors_HaveTier(t *testing.T) {
	registry := NewFactory().CreateAll()

	expectedTiers := map[SelectionMethod]AlgorithmTier{
		MethodStatic:      TierSupported,
		MethodRandom:      TierSupported,
		MethodMultiFactor: TierSupported,
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
	SetGlobalRegistry(NewFactory().CreateAll())

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
