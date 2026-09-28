package classification

import "testing"

// These margins exercise the shipped operating points and decision wiring;
// they are controlled inputs, not embedding-model quality measurements.
func TestBuiltinComplexityKeepsIndependentBoundariesAndMiddle(t *testing.T) {
	for _, profile := range []struct{ name, ordinary, easy string }{
		{"balance", "medium", "simple"},
		{"speed", "fast", "fast"},
		{"cost", "economy", "economy"},
		{"accuracy", "simple", "simple"},
	} {
		t.Run(profile.name, func(t *testing.T) {
			c := builtinPolicyClassifier(t, profile.name)
			rule := c.Config.ComplexityRules[0]
			if rule.Name != "difficulty" || rule.Threshold != 0 || rule.HardAbove == nil || rule.EasyBelow == nil {
				t.Fatal("difficulty must use an explicit pair instead of a symmetric threshold")
			}
			bounds, err := rule.EffectiveBoundaries()
			if err != nil {
				t.Fatal(err)
			}
			for _, tt := range []struct {
				margin float64
				label  string
			}{
				{-.080001, "easy"},
				{-.08, "medium"},
				{-.03, "medium"},
				{0, "medium"},
				{.025, "medium"},
				{.025001, "hard"},
			} {
				label := localComplexityVerdict(bounds, tt.margin)
				if label != tt.label {
					t.Fatalf("margin %g: got %s, want %s", tt.margin, label, tt.label)
				}
				want := profile.ordinary
				switch label {
				case "easy":
					want = profile.easy
				case "hard":
					want = "reasoning"
				}
				assertBuiltinPolicy(t, c, &SignalResults{MatchedComplexityRules: []string{"difficulty:" + label}}, want)
			}
		})
	}
}
