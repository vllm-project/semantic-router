package config

import (
	"strings"
	"testing"
)

func TestPolarityGuardConfigDefaults(t *testing.T) {
	var nilGuard *PolarityGuardConfig
	if got := nilGuard.NormalizedMode(); got != PolarityGuardModeLexical {
		t.Fatalf("nil guard mode = %q, want %q", got, PolarityGuardModeLexical)
	}
	if nilGuard.UsesNLI() {
		t.Fatal("nil guard must not enable the NLI tier")
	}
	if got := nilGuard.EffectiveContradictionThreshold(); got != DefaultPolarityContradictionThreshold {
		t.Fatalf("nil guard threshold = %v, want %v", got, DefaultPolarityContradictionThreshold)
	}

	cases := []struct {
		mode     string
		wantMode string
		wantNLI  bool
	}{
		{"", PolarityGuardModeLexical, false},
		{"lexical", PolarityGuardModeLexical, false},
		{"  NLI ", PolarityGuardModeNLI, true},
		{"lexical+nli", PolarityGuardModeLexicalNLI, true},
		{"Lexical+NLI", PolarityGuardModeLexicalNLI, true},
	}
	for _, tc := range cases {
		guard := &PolarityGuardConfig{Mode: tc.mode}
		if got := guard.NormalizedMode(); got != tc.wantMode {
			t.Errorf("mode %q normalized = %q, want %q", tc.mode, got, tc.wantMode)
		}
		if got := guard.UsesNLI(); got != tc.wantNLI {
			t.Errorf("mode %q UsesNLI = %v, want %v", tc.mode, got, tc.wantNLI)
		}
	}

	custom := &PolarityGuardConfig{NLI: PolarityGuardNLIConfig{ContradictionThreshold: f32(0.7)}}
	if got := custom.EffectiveContradictionThreshold(); got != 0.7 {
		t.Fatalf("explicit threshold = %v, want 0.7", got)
	}
}

func TestValidatePolarityGuard(t *testing.T) {
	withGuard := func(enabled bool, guard *PolarityGuardConfig, nliModel string) *RouterConfig {
		cfg := &RouterConfig{}
		cfg.SemanticCache.Enabled = enabled
		cfg.SemanticCache.PolarityGuard = guard
		cfg.HallucinationMitigation.NLIModel.ModelID = nliModel
		return cfg
	}

	cases := []struct {
		name    string
		cfg     *RouterConfig
		wantErr string
	}{
		{"absent_block_ok", withGuard(true, nil, ""), ""},
		{"lexical_without_model_ok", withGuard(true, &PolarityGuardConfig{Mode: "lexical"}, ""), ""},
		{"nli_with_model_ok", withGuard(true, &PolarityGuardConfig{Mode: "nli"}, "models/mom-halugate-explainer"), ""},
		{"lexical_nli_with_model_ok", withGuard(true, &PolarityGuardConfig{Mode: "lexical+nli"}, "models/mom-halugate-explainer"), ""},
		{"nli_without_model_rejected", withGuard(true, &PolarityGuardConfig{Mode: "nli"}, ""), "hallucination_mitigation.explainer"},
		{"nli_without_model_on_disabled_cache_ok", withGuard(false, &PolarityGuardConfig{Mode: "nli"}, ""), ""},
		{"unknown_mode_rejected", withGuard(true, &PolarityGuardConfig{Mode: "cross-encoder"}, "models/x"), `mode must be one of`},
		{"threshold_above_one_rejected", withGuard(true, &PolarityGuardConfig{
			Mode: "nli", NLI: PolarityGuardNLIConfig{ContradictionThreshold: f32(1.5)},
		}, "models/x"), "contradiction_threshold"},
		{"threshold_negative_rejected", withGuard(true, &PolarityGuardConfig{
			Mode: "lexical", NLI: PolarityGuardNLIConfig{ContradictionThreshold: f32(-0.1)},
		}, ""), "contradiction_threshold"},
		{"threshold_bounds_ok", withGuard(true, &PolarityGuardConfig{
			Mode: "nli", NLI: PolarityGuardNLIConfig{ContradictionThreshold: f32(1.0)},
		}, "models/x"), ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := validateSemanticCacheContracts(tc.cfg)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error %q does not mention %q", err.Error(), tc.wantErr)
			}
		})
	}
}

func TestNeedsLocalNLIForSemanticCache(t *testing.T) {
	build := func(enabled bool, mode string, nliModel string) *RouterConfig {
		cfg := &RouterConfig{}
		cfg.SemanticCache.Enabled = enabled
		if mode != "" {
			cfg.SemanticCache.PolarityGuard = &PolarityGuardConfig{Mode: mode}
		}
		cfg.HallucinationMitigation.NLIModel.ModelID = nliModel
		return cfg
	}

	var nilCfg *RouterConfig
	if nilCfg.NeedsLocalNLIForSemanticCache() {
		t.Fatal("nil config must not require the NLI model")
	}

	cases := []struct {
		name string
		cfg  *RouterConfig
		want bool
	}{
		{"nli_mode_enabled_cache_with_model", build(true, "nli", "models/mom-halugate-explainer"), true},
		{"lexical_nli_mode_enabled_cache_with_model", build(true, "lexical+nli", "models/mom-halugate-explainer"), true},
		{"lexical_mode", build(true, "lexical", "models/mom-halugate-explainer"), false},
		{"absent_guard", build(true, "", "models/mom-halugate-explainer"), false},
		{"disabled_cache", build(false, "nli", "models/mom-halugate-explainer"), false},
		{"no_model_configured", build(true, "nli", ""), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.cfg.NeedsLocalNLIForSemanticCache(); got != tc.want {
				t.Fatalf("NeedsLocalNLIForSemanticCache = %v, want %v", got, tc.want)
			}
		})
	}

	// The NLI tier never requires the hallucination plugin: the explainer model
	// is bound for the cache even when hallucination mitigation is off.
	cfg := build(true, "nli", "models/mom-halugate-explainer")
	if cfg.NeedsLocalHallucinationNLIForRouting() || cfg.NeedsLocalHallucinationNLIForAPI() {
		t.Fatal("cache-only NLI must not be reported as a hallucination consumer")
	}
}
