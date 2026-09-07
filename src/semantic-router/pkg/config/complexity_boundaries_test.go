package config

import (
	"strings"
	"testing"
)

// threshold: X keeps working as the symmetric shorthand, because the local
// margin is signed and centred on zero: anything above +X is hard, anything
// below -X is easy, and the band between them is medium.
func TestComplexityBoundaries_ThresholdIsSymmetricShorthand(t *testing.T) {
	bounds, err := ComplexityRule{Threshold: 0.10}.EffectiveBoundaries()
	if err != nil {
		t.Fatalf("EffectiveBoundaries: %v", err)
	}

	// Threshold is float32 on the existing schema, so widening it cannot be
	// exact; the shorthand only has to stay symmetric about zero.
	if !closeEnough(bounds.HardAt, 0.10) || !closeEnough(bounds.EasyAt, -0.10) {
		t.Fatalf("boundaries = %+v, want hard +0.10 / easy -0.10", bounds)
	}
	if !bounds.HigherIsHarder {
		t.Error("the symmetric shorthand must keep the local higher-is-harder direction")
	}
}

// A higher-is-harder remote model names its boundaries with the pair that
// reads in that direction.
func TestComplexityBoundaries_HigherIsHarder(t *testing.T) {
	hardAbove, easyBelow := 0.85, 0.60
	bounds, err := ComplexityRule{HardAbove: &hardAbove, EasyBelow: &easyBelow}.EffectiveBoundaries()
	if err != nil {
		t.Fatalf("EffectiveBoundaries: %v", err)
	}

	if !bounds.HigherIsHarder {
		t.Error("hard_above/easy_below must resolve to higher-is-harder")
	}
	if bounds.HardAt != 0.85 || bounds.EasyAt != 0.60 {
		t.Fatalf("boundaries = %+v, want hard 0.85 / easy 0.60", bounds)
	}
}

// A model whose score falls as difficulty rises - a predicted-success-rate
// scorer, say - uses the opposite pair, and the direction comes from the field
// names rather than a separate setting.
func TestComplexityBoundaries_LowerIsHarder(t *testing.T) {
	hardBelow, easyAbove := 0.40, 0.80
	bounds, err := ComplexityRule{HardBelow: &hardBelow, EasyAbove: &easyAbove}.EffectiveBoundaries()
	if err != nil {
		t.Fatalf("EffectiveBoundaries: %v", err)
	}

	if bounds.HigherIsHarder {
		t.Error("hard_below/easy_above must resolve to lower-is-harder")
	}
	if bounds.HardAt != 0.40 || bounds.EasyAt != 0.80 {
		t.Fatalf("boundaries = %+v, want hard 0.40 / easy 0.80", bounds)
	}
}

// The bands must not overlap: a score that satisfies both verdicts has no
// defined answer, so it is rejected at config load rather than resolved by
// evaluation order.
func TestComplexityBoundaries_RejectsOverlappingBands(t *testing.T) {
	cases := map[string]ComplexityRule{
		"higher-is-harder overlap": {
			HardAbove: floatPtr(0.60),
			EasyBelow: floatPtr(0.85),
		},
		"lower-is-harder overlap": {
			HardBelow: floatPtr(0.80),
			EasyAbove: floatPtr(0.40),
		},
	}

	for name, rule := range cases {
		if _, err := rule.EffectiveBoundaries(); err == nil {
			t.Errorf("%s: expected an overlapping band to be rejected", name)
		}
	}
}

// Mixing the two pairs states two directions at once.
func TestComplexityBoundaries_RejectsMixedDirections(t *testing.T) {
	rule := ComplexityRule{
		HardAbove: floatPtr(0.85),
		EasyAbove: floatPtr(0.80),
	}

	_, err := rule.EffectiveBoundaries()
	if err == nil {
		t.Fatal("expected a rule mixing hard_above with easy_above to be rejected")
	}
	if !strings.Contains(err.Error(), "direction") {
		t.Errorf("error %q should explain that the pair states two directions", err.Error())
	}
}

// Declaring only one side leaves the other verdict unreachable, which is
// almost certainly a mistake rather than an intent.
func TestComplexityBoundaries_RejectsHalfDeclaredPair(t *testing.T) {
	cases := map[string]ComplexityRule{
		"hard_above alone": {HardAbove: floatPtr(0.85)},
		"easy_below alone": {EasyBelow: floatPtr(0.60)},
		"hard_below alone": {HardBelow: floatPtr(0.40)},
		"easy_above alone": {EasyAbove: floatPtr(0.80)},
	}

	for name, rule := range cases {
		if _, err := rule.EffectiveBoundaries(); err == nil {
			t.Errorf("%s: expected a half-declared pair to be rejected", name)
		}
	}
}

// An explicit pair and the shorthand say the same thing two ways.
func TestComplexityBoundaries_RejectsThresholdAlongsidePair(t *testing.T) {
	rule := ComplexityRule{
		Threshold: 0.10,
		HardAbove: floatPtr(0.85),
		EasyBelow: floatPtr(0.60),
	}

	if _, err := rule.EffectiveBoundaries(); err == nil {
		t.Fatal("expected threshold alongside an explicit boundary pair to be rejected")
	}
}

func closeEnough(got, want float64) bool {
	const tolerance = 1e-6
	diff := got - want
	if diff < 0 {
		diff = -diff
	}
	return diff < tolerance
}
