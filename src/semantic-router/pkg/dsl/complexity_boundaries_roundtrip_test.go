package dsl

import (
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

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

// EmitYAML is the public DSL-to-YAML path, and it must not launder a rule the
// loader refuses. Threshold carries omitempty so a pair rule does not grow a
// threshold: 0 on emission, which means a written zero has to be preserved
// deliberately: dropped, the emitted document reloads with the threshold
// absent and the same pair the compiler recorded as refused is accepted.
func TestEmitYAMLKeepsWrittenZeroThresholdRefusedOnReload(t *testing.T) {
	cases := map[string]struct {
		source  string
		refused bool
	}{
		"written zero beside a pair": {
			source:  "SIGNAL complexity r {\n  threshold: 0\n  hard_above: 0.85\n  easy_below: 0.60\n}",
			refused: true,
		},
		"pair with no threshold": {
			source:  "SIGNAL complexity r {\n  hard_above: 0.85\n  easy_below: 0.60\n}",
			refused: false,
		},
	}

	for name, tc := range cases {
		emitted, errs := EmitYAML(tc.source)
		if len(errs) > 0 {
			t.Fatalf("%s: EmitYAML: %v", name, errs)
		}

		// The emitted document decodes with presence intact.
		var canonical config.CanonicalConfig
		if err := yaml.Unmarshal(emitted, &canonical); err != nil {
			t.Fatalf("%s: reload emitted YAML: %v", name, err)
		}
		if len(canonical.Routing.Signals.Complexity) != 1 {
			t.Fatalf("%s: reloaded %d complexity rules, want 1", name, len(canonical.Routing.Signals.Complexity))
		}
		rule := canonical.Routing.Signals.Complexity[0]
		if rule.ThresholdSet != tc.refused {
			t.Errorf("%s: ThresholdSet after reload = %v, want %v", name, rule.ThresholdSet, tc.refused)
		}

		// And the loader reaches the same verdict the compiler did.
		_, err := config.ParseYAMLBytes(emitted)
		switch {
		case tc.refused && (err == nil || !strings.Contains(err.Error(), "keep one")):
			t.Errorf("%s: expected the reloaded document to be refused for stating both, got %v", name, err)
		case !tc.refused && err != nil:
			t.Errorf("%s: a pair with no threshold must reload cleanly: %v", name, err)
		}
	}
}
