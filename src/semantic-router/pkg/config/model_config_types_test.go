package config

import "testing"

func TestPrototypeScoringConfigWithDefaultsEnablesPrototypeScoring(t *testing.T) {
	cfg := PrototypeScoringConfig{}.WithDefaults()
	if cfg.Enabled == nil || !*cfg.Enabled {
		t.Fatal("expected prototype scoring to be enabled by default")
	}
	if cfg.ClusterSimilarityThreshold <= 0 {
		t.Fatal("expected default cluster_similarity_threshold to be positive")
	}
	if cfg.MaxPrototypes <= 0 {
		t.Fatal("expected default max_prototypes to be positive")
	}
}
