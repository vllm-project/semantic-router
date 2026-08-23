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
		if entry.Execution != AlgorithmExecutionSelector && entry.Execution != AlgorithmExecutionLooper {
			t.Errorf("Catalog entry %q has invalid Execution %q", entry.Type, entry.Execution)
		}
	}
}

func TestSupportedLooperAlgorithmTypes(t *testing.T) {
	want := []string{
		DecisionAlgorithmConfidence,
		DecisionAlgorithmFusion,
		DecisionAlgorithmRatings,
		DecisionAlgorithmReMoM,
		DecisionAlgorithmWorkflows,
	}
	got := SupportedLooperAlgorithmTypes()

	if len(got) != len(want) {
		t.Fatalf("SupportedLooperAlgorithmTypes() = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("SupportedLooperAlgorithmTypes() = %v, want %v", got, want)
		}
		if !IsLooperAlgorithmType(got[i]) {
			t.Errorf("IsLooperAlgorithmType(%q) = false, want true", got[i])
		}
		if !IsSupportedDecisionAlgorithmType(got[i]) {
			t.Errorf("Looper algorithm %q is missing from the public decision catalog", got[i])
		}
	}

	for _, algorithmType := range []string{DecisionAlgorithmStatic, "rl_driven", "unknown"} {
		if IsLooperAlgorithmType(algorithmType) {
			t.Errorf("IsLooperAlgorithmType(%q) = true, want false", algorithmType)
		}
	}
}

func TestDecisionAlgorithmDispatchCardinalityIsExhaustiveAndClosed(t *testing.T) {
	cardinality, ok := DecisionAlgorithmDispatchCardinality("")
	if !ok || cardinality != AlgorithmDispatchSingle {
		t.Fatalf("default dispatch cardinality = %q, %v", cardinality, ok)
	}
	for _, entry := range DecisionAlgorithmCatalog() {
		cardinality, ok := DecisionAlgorithmDispatchCardinality(entry.Type)
		if !ok {
			t.Fatalf("catalog algorithm %q has no dispatch cardinality", entry.Type)
		}
		want := AlgorithmDispatchSingle
		if entry.Execution == AlgorithmExecutionLooper {
			want = AlgorithmDispatchMulti
		}
		if cardinality != want {
			t.Errorf("algorithm %q cardinality = %q, want %q", entry.Type, cardinality, want)
		}
	}
	if cardinality, ok := DecisionAlgorithmDispatchCardinality("unknown"); ok || cardinality != "" {
		t.Fatalf("unknown algorithm cardinality = %q, %v; want closed rejection", cardinality, ok)
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
