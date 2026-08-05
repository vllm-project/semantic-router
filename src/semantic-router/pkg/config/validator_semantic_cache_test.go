package config

import (
	"strings"
	"testing"
)

func f32(v float32) *float32 { return &v }

func TestValidateSemanticCacheThresholdRange(t *testing.T) {
	cases := []struct {
		name      string
		threshold *float32
		wantErr   bool
	}{
		{"unset_ok", nil, false},
		{"zero_ok", f32(0), false},
		{"mid_ok", f32(0.8), false},
		{"one_ok", f32(1.0), false},
		{"above_one_rejected", f32(1.5), true},
		{"negative_rejected", f32(-0.1), true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &RouterConfig{}
			cfg.SemanticCache.SimilarityThreshold = tc.threshold
			err := validateSemanticCacheContracts(cfg)
			if tc.wantErr != (err != nil) {
				t.Fatalf("threshold=%v: wantErr=%v, got err=%v", tc.threshold, tc.wantErr, err)
			}
		})
	}
}

func TestValidateSemanticCachePerDecisionThreshold(t *testing.T) {
	mkDecision := func(name string, threshold float32) Decision {
		return Decision{
			Name: name,
			Plugins: []DecisionPlugin{{
				Type: "semantic-cache",
				Configuration: MustStructuredPayload(map[string]interface{}{
					"enabled":              true,
					"similarity_threshold": threshold,
				}),
			}},
		}
	}

	// Valid per-decision threshold passes.
	okCfg := &RouterConfig{}
	okCfg.Decisions = []Decision{mkDecision("route_ok", 0.7)}
	if err := validateSemanticCacheContracts(okCfg); err != nil {
		t.Fatalf("valid per-decision threshold rejected: %v", err)
	}

	// Out-of-range per-decision threshold is rejected and names the decision.
	badCfg := &RouterConfig{}
	badCfg.Decisions = []Decision{mkDecision("route_bad", 2.0)}
	err := validateSemanticCacheContracts(badCfg)
	if err == nil {
		t.Fatal("out-of-range per-decision threshold must be rejected")
	}
	if !strings.Contains(err.Error(), "route_bad") {
		t.Fatalf("error should name the offending decision, got: %v", err)
	}
}

func TestValidateSemanticCacheNilConfig(t *testing.T) {
	if err := validateSemanticCacheContracts(nil); err != nil {
		t.Fatalf("nil config must be valid, got: %v", err)
	}
}
