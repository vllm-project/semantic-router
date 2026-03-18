package config

import "testing"

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
