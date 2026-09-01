package config

import "testing"

func TestHNSWConfigWithDefaultsDefersMultimodalDimension(t *testing.T) {
	for _, modelType := range []string{"multimodal", " MultiModal "} {
		cfg := HNSWConfig{ModelType: modelType}.WithDefaults()
		if cfg.TargetDimension != 0 {
			t.Fatalf("multimodal dimension must be resolved from the loaded model, got %d", cfg.TargetDimension)
		}
	}
}

func TestPreferenceModelConfigWithDefaultsEnablesContrastiveByDefault(t *testing.T) {
	cfg := PreferenceModelConfig{}.WithDefaults()
	if cfg.UseContrastive == nil || !*cfg.UseContrastive {
		t.Fatal("expected default preference config to enable contrastive mode")
	}
}

func TestPreferenceModelConfigWithDefaultsPreservesExplicitFalse(t *testing.T) {
	disabled := false
	cfg := PreferenceModelConfig{UseContrastive: &disabled}.WithDefaults()
	if cfg.UseContrastive == nil {
		t.Fatal("expected explicit false preference config to be preserved")
	}
	if *cfg.UseContrastive {
		t.Fatal("expected explicit false preference config to remain disabled")
	}
}

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

func TestComplexityModelConfigWithDefaultsEnablesPrototypeScoring(t *testing.T) {
	cfg := ComplexityModelConfig{}.WithDefaults()
	if cfg.PrototypeScoring.Enabled == nil || !*cfg.PrototypeScoring.Enabled {
		t.Fatal("expected complexity prototype scoring to inherit default enablement")
	}
}
