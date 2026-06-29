package config

import "testing"

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

func TestDecisionAlgorithmCatalog_PublicAlgorithmSurface(t *testing.T) {
	publicTypes := []string{
		"automix", "confidence", "fusion", "hybrid", "kmeans",
		"knn", "latency_aware", "mlp", "multi_factor", "ratings",
		"remom", "router_dc", "static", "svm",
	}

	for _, algType := range publicTypes {
		if !IsSupportedDecisionAlgorithmType(algType) {
			t.Errorf("public algorithm type %q is not supported", algType)
		}
	}
	for _, migratedType := range []string{"elo", "rl_driven", "gmtrouter", "session_aware"} {
		if IsSupportedDecisionAlgorithmType(migratedType) {
			t.Errorf("learning-owned algorithm type %q should not be public", migratedType)
		}
	}
}

func TestGetAlgorithmTier(t *testing.T) {
	tests := []struct {
		algType      string
		expectedTier string
	}{
		{"static", "supported"},
		{"router_dc", "supported"},
		{"latency_aware", "supported"},
		{"hybrid", "supported"},
		{"automix", "experimental"},
		{"fusion", "experimental"},
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
