package config

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
)

func TestCacheEconomicsEffectiveConfigAndValidation(t *testing.T) {
	off := false
	for _, tc := range []struct {
		name   string
		tuning *CacheEconomicsTuning
		want   string
	}{
		{"defaults", nil, ""},
		{"disabled", &CacheEconomicsTuning{Enabled: &off}, ""},
		{"one_hour", &CacheEconomicsTuning{CacheTTLSeconds: intPtr(3600), DefaultRemainingTurns: intPtr(8)}, ""},
		{"no_lifetime", &CacheEconomicsTuning{CacheTTLSeconds: intPtr(0)}, "cache_ttl_seconds"},
		{"past_a_day", &CacheEconomicsTuning{CacheTTLSeconds: intPtr(86401)}, "cache_ttl_seconds"},
		{"no_turns", &CacheEconomicsTuning{DefaultRemainingTurns: intPtr(0)}, "default_remaining_turns"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := validateProtectionTuning("tuning", RouterLearningProtectionTuning{CacheEconomics: tc.tuning})
			if tc.want == "" && err != nil || tc.want != "" && (err == nil || !strings.Contains(err.Error(), tc.want)) {
				t.Fatalf("error = %v, want %q", err, tc.want)
			}
		})
	}
	cfg := (*CacheEconomicsTuning)(nil).EffectiveConfig()
	if !cfg.Enabled || cfg.CacheTTLSeconds != 300 || cfg.DefaultRemainingTurns != 3 {
		t.Fatalf("defaults changed: %+v", cfg)
	}
	if (&CacheEconomicsTuning{Enabled: &off}).EffectiveConfig().Enabled {
		t.Fatal("enabled: false must turn the policy off")
	}
}

func TestCacheEconomicsTuningIsAcceptedUnderProtection(t *testing.T) {
	document := func(field string) map[string]interface{} {
		var raw map[string]interface{}
		if err := yaml.Unmarshal([]byte(`
global:
  router:
    learning:
      protection:
        tuning:
          cache_economics: {enabled: true, cache_ttl_seconds: 3600, `+field+`: 5}
`), &raw); err != nil {
			t.Fatalf("yaml: %v", err)
		}
		return raw
	}
	if err := rejectUnsupportedRouterLearningFields(document("default_remaining_turns")); err != nil {
		t.Fatalf("cache_economics must be accepted: %v", err)
	}
	if err := rejectUnsupportedRouterLearningFields(document("remaining_turns")); err == nil ||
		!strings.Contains(err.Error(), "cache_economics.remaining_turns") {
		t.Fatalf("an unknown cache_economics field must be rejected, got %v", err)
	}
}
