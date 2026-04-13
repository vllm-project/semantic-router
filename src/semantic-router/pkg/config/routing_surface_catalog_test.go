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

func TestDecisionAlgorithmCatalog_BackwardsCompatible(t *testing.T) {
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
