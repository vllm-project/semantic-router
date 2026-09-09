package dsl

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A rule's declared cut points must survive YAML -> DSL -> YAML. Dropping them
// silently reverts the rule to threshold semantics, where threshold is zero:
// every positive margin becomes hard and medium is unreachable, while the
// config still reads as though the boundaries applied.
func TestComplexityBoundaryPairSurvivesARoundTrip(t *testing.T) {
	cases := map[string]config.ComplexityRule{
		"higher is harder": {
			Name:      "needs_reasoning",
			HardAbove: float64Ptr(0.85),
			EasyBelow: float64Ptr(0.60),
		},
		"lower is harder": {
			Name:      "predicted_success",
			HardBelow: float64Ptr(0.40),
			EasyAbove: float64Ptr(0.80),
		},
		"threshold shorthand": {
			Name:      "classic",
			Threshold: 0.10,
		},
	}

	for name, rule := range cases {
		original := &config.RouterConfig{}
		original.ComplexityRules = []config.ComplexityRule{rule}

		restored := roundTripComplexityRules(t, original)
		if len(restored) != 1 {
			t.Fatalf("%s: restored %d rules, want 1", name, len(restored))
		}
		assertSameBoundaries(t, name, rule, restored[0])
	}
}

func roundTripComplexityRules(t *testing.T, cfg *config.RouterConfig) []config.ComplexityRule {
	t.Helper()
	source, err := Decompile(cfg)
	if err != nil {
		t.Fatalf("Decompile: %v", err)
	}
	restored, errs := Compile(source)
	if len(errs) > 0 {
		t.Fatalf("Compile: %v\n--- generated source ---\n%s", errs, source)
	}
	return restored.ComplexityRules
}

func assertSameBoundaries(t *testing.T, name string, want, got config.ComplexityRule) {
	t.Helper()
	wantBounds, err := want.EffectiveBoundaries()
	if err != nil {
		t.Fatalf("%s: boundaries for the original rule: %v", name, err)
	}
	gotBounds, err := got.EffectiveBoundaries()
	if err != nil {
		t.Fatalf("%s: boundaries after the round trip: %v", name, err)
	}
	if wantBounds != gotBounds {
		t.Errorf("%s: boundaries after round trip = %+v, want %+v", name, gotBounds, wantBounds)
	}
}
