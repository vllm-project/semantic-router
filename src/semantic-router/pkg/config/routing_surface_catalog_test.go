package config

import (
	"encoding/json"
	"os"
	"testing"
)

type selectorCoverageEntry struct {
	Algorithm string `json:"algorithm"`
	Tier      string `json:"tier"`
	Status    string `json:"status"`
}

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

func TestSelectorAlgorithmCoverageTracksRuntimeCatalog(t *testing.T) {
	raw, err := os.ReadFile("../../../../e2e/pkg/testcases/testdata/selector_algorithm_coverage.json")
	if err != nil {
		t.Fatal(err)
	}
	var coverage []selectorCoverageEntry
	if err := json.Unmarshal(raw, &coverage); err != nil {
		t.Fatal(err)
	}

	byAlgorithm := make(map[string]selectorCoverageEntry, len(coverage))
	for _, entry := range coverage {
		if entry.Algorithm == "" {
			t.Fatal("selector coverage entry has empty algorithm")
		}
		if _, exists := byAlgorithm[entry.Algorithm]; exists {
			t.Errorf("selector coverage has duplicate entry for %q", entry.Algorithm)
		}
		byAlgorithm[entry.Algorithm] = entry
	}

	selectorCount := 0
	for _, algorithm := range DecisionAlgorithmCatalog() {
		if algorithm.Execution != AlgorithmExecutionSelector {
			continue
		}
		selectorCount++
		entry, ok := byAlgorithm[algorithm.Type]
		if !ok {
			t.Errorf("selector algorithm %q has no E2E coverage entry", algorithm.Type)
			continue
		}
		if entry.Tier != algorithm.Tier {
			t.Errorf("selector algorithm %q coverage tier = %q, runtime tier = %q", algorithm.Type, entry.Tier, algorithm.Tier)
		}
		delete(byAlgorithm, algorithm.Type)
	}

	if len(coverage) != selectorCount {
		t.Errorf("selector coverage has %d entries, runtime catalog has %d selectors", len(coverage), selectorCount)
	}
	for algorithm := range byAlgorithm {
		t.Errorf("selector coverage entry %q is absent from the runtime selector catalog", algorithm)
	}
}
