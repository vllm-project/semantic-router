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

// The DSL is the other way a rule is written, so it records presence the way
// the YAML loader does: a written `threshold: 0` next to a pair is refused
// here too, rather than compiling into a rule the Router accepts and the CRD
// does not.
func TestCompileComplexitySignal_RecordsWrittenZeroThreshold(t *testing.T) {
	cfg, errs := Compile(`SIGNAL complexity needs_reasoning {
  threshold: 0
  hard_above: 0.85
  easy_below: 0.60
}`)
	if len(errs) > 0 {
		t.Fatalf("compile errors: %v", errs)
	}
	if len(cfg.ComplexityRules) != 1 {
		t.Fatalf("compiled %d complexity rules, want 1", len(cfg.ComplexityRules))
	}
	rule := cfg.ComplexityRules[0]
	if !rule.ThresholdSet {
		t.Error("a written threshold must be recorded as present")
	}
	if _, err := rule.EffectiveBoundaries(); err == nil {
		t.Error("expected a written threshold: 0 alongside a pair to be refused")
	}
}

// A written zero survives the round trip as written. Dropping it would turn a
// stated threshold into an absent one, and the two are no longer the same
// thing once presence is part of the verdict.
func TestComplexityWrittenZeroThresholdSurvivesARoundTrip(t *testing.T) {
	original := &config.RouterConfig{}
	original.ComplexityRules = []config.ComplexityRule{{Name: "classic", ThresholdSet: true}}

	restored := roundTripComplexityRules(t, original)
	if len(restored) != 1 {
		t.Fatalf("restored %d rules, want 1", len(restored))
	}
	if !restored[0].ThresholdSet || restored[0].Threshold != 0 {
		t.Fatalf("written zero threshold did not survive: %#v", restored[0])
	}
}
