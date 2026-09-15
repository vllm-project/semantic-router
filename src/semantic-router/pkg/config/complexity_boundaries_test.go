package config

import (
	"math"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
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

// A non-finite boundary cannot separate anything: every comparison against
// NaN is false, so a rule carrying one silently answers medium for every
// score, and an infinite cut point makes one verdict unreachable.
func TestComplexityBoundaries_RejectsNonFiniteValues(t *testing.T) {
	nan := math.NaN()
	posInf := math.Inf(1)
	negInf := math.Inf(-1)

	cases := map[string]ComplexityRule{
		"NaN hard_above":  {Name: "r", HardAbove: &nan, EasyBelow: floatPtr(0.1)},
		"NaN easy_below":  {Name: "r", HardAbove: floatPtr(0.9), EasyBelow: &nan},
		"+Inf hard_above": {Name: "r", HardAbove: &posInf, EasyBelow: floatPtr(0.1)},
		"-Inf easy_below": {Name: "r", HardAbove: floatPtr(0.9), EasyBelow: &negInf},
		"NaN hard_below":  {Name: "r", HardBelow: &nan, EasyAbove: floatPtr(0.9)},
		"+Inf easy_above": {Name: "r", HardBelow: floatPtr(0.1), EasyAbove: &posInf},
	}

	for name, rule := range cases {
		if _, err := rule.EffectiveBoundaries(); err == nil {
			t.Errorf("%s: expected a non-finite boundary to be rejected", name)
		}
	}
}

// threshold expands into the same pair the explicit fields declare, so it
// needs the same two guards. A non-finite value makes every comparison false
// and the rule answers medium for everything; a negative one puts hard at a
// negative cut and easy at a positive one, so Verdict's hard test is true for
// nearly every score and easy is unreachable - the rule reads as "escalate a
// little" and behaves as "escalate everything".
func TestComplexityBoundaries_RejectsUnusableThreshold(t *testing.T) {
	cases := map[string]ComplexityRule{
		"NaN threshold":      {Name: "r", Threshold: float32(math.NaN())},
		"+Inf threshold":     {Name: "r", Threshold: float32(math.Inf(1))},
		"-Inf threshold":     {Name: "r", Threshold: float32(math.Inf(-1))},
		"negative threshold": {Name: "r", Threshold: -0.1},
	}

	for name, rule := range cases {
		if _, err := rule.EffectiveBoundaries(); err == nil {
			t.Errorf("%s: expected an unusable threshold to be rejected", name)
		}
	}
}

// The shapes that must keep working: a positive threshold, and an omitted one
// (which collapses both cut points onto zero, the long-standing default for a
// rule that states no threshold at all).
func TestComplexityBoundaries_AcceptsUsableThreshold(t *testing.T) {
	cases := map[string]ComplexityRule{
		"positive": {Name: "r", Threshold: 0.1},
		"omitted":  {Name: "r"},
	}

	for name, rule := range cases {
		bounds, err := rule.EffectiveBoundaries()
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		// Widened through float64 exactly as EffectiveBoundaries does: a
		// float32 0.1 is 0.10000000149011612 as a float64, so comparing
		// against the literal would fail on precision rather than on
		// behaviour.
		want := float64(rule.Threshold)
		if bounds.HardAt != want || bounds.EasyAt != -want || !bounds.HigherIsHarder {
			t.Errorf("%s: boundaries = %+v, want hard %v / easy %v, higher-is-harder",
				name, bounds, want, -want)
		}
	}
}

// The CRD refuses `threshold` written alongside a boundary pair whatever its
// value, through has(self.threshold). Threshold is a float32 with no presence
// of its own, so a written zero used to load as an absent key and the same
// YAML was rejected by kubectl apply yet accepted by the Router. Presence is
// now recorded on load and the two agree.
func TestComplexityBoundaries_RejectsWrittenZeroThresholdAlongsidePair(t *testing.T) {
	cases := map[string]string{
		"higher is harder": "name: r\nthreshold: 0\nhard_above: 0.85\neasy_below: 0.60\n",
		"lower is harder":  "name: r\nthreshold: 0\nhard_below: 0.40\neasy_above: 0.80\n",
		"zero as a float":  "name: r\nthreshold: 0.0\nhard_above: 0.85\neasy_below: 0.60\n",
	}

	for name, doc := range cases {
		var rule ComplexityRule
		if err := yaml.Unmarshal([]byte(doc), &rule); err != nil {
			t.Fatalf("%s: unmarshal: %v", name, err)
		}
		if !rule.ThresholdSet {
			t.Errorf("%s: a written threshold must be recorded as present", name)
		}
		_, err := rule.EffectiveBoundaries()
		if err == nil || !strings.Contains(err.Error(), "keep one") {
			t.Errorf("%s: expected a written threshold: 0 alongside a pair to be refused, got %v", name, err)
		}
	}
}

// Presence is about the key, not the value: an omitted key and a null one are
// absent, and a written value of any size is present. A written zero on its
// own keeps meaning the shorthand it always did.
func TestComplexityRule_ThresholdPresenceFollowsTheKey(t *testing.T) {
	cases := map[string]struct {
		doc     string
		present bool
	}{
		"omitted":  {doc: "name: r\n", present: false},
		"null":     {doc: "name: r\nthreshold: null\n", present: false},
		"zero":     {doc: "name: r\nthreshold: 0\n", present: true},
		"positive": {doc: "name: r\nthreshold: 0.1\n", present: true},
	}

	for name, tc := range cases {
		var rule ComplexityRule
		if err := yaml.Unmarshal([]byte(tc.doc), &rule); err != nil {
			t.Fatalf("%s: unmarshal: %v", name, err)
		}
		if rule.ThresholdSet != tc.present {
			t.Errorf("%s: ThresholdSet = %v, want %v", name, rule.ThresholdSet, tc.present)
		}
		bounds, err := rule.EffectiveBoundaries()
		if err != nil {
			t.Errorf("%s: a threshold without a pair must still resolve: %v", name, err)
			continue
		}
		want := float64(rule.Threshold)
		if bounds.HardAt != want || bounds.EasyAt != -want || !bounds.HigherIsHarder {
			t.Errorf("%s: boundaries = %+v, want the symmetric shorthand at %v", name, bounds, want)
		}
	}
}

// A rule that states a pair is written back out by the operator and the DSL
// emitter. It must not grow a `threshold: 0` on the way, or the reloaded
// document would be refused for stating both.
func TestComplexityRule_ZeroThresholdIsNotWrittenBack(t *testing.T) {
	rule := ComplexityRule{Name: "r", HardAbove: floatPtr(0.85), EasyBelow: floatPtr(0.60)}

	out, err := yaml.Marshal(rule)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(out), "threshold") {
		t.Fatalf("a zero threshold must be omitted on marshal, got:\n%s", out)
	}

	var reloaded ComplexityRule
	if err := yaml.Unmarshal(out, &reloaded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if reloaded.ThresholdSet {
		t.Error("a marshalled pair rule must reload without a threshold")
	}
	if _, err := reloaded.EffectiveBoundaries(); err != nil {
		t.Fatalf("a pair rule must survive a marshal round trip: %v", err)
	}
}
