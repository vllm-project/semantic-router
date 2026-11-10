package config

import "testing"

// Masking asks the pii rule it names although no decision reads the rule, so
// the signal-type walk alone left the rule's deployment unprepared, and
// because masking fails closed every request on a masking-only recipe
// answered 503. Found by the first cluster run of the masking e2e profile
// (#3806 review).
func TestMaskingPluginAloneAsksItsPIIRule(t *testing.T) {
	masking := func(enabled bool) Decision {
		return Decision{
			Name:      "mask",
			ModelRefs: []ModelRef{{Model: "m"}},
			Plugins: []DecisionPlugin{{
				Type:          DecisionPluginMasking,
				Configuration: MustStructuredPayload(map[string]interface{}{"enabled": enabled, "signal": "personal"}),
			}},
		}
	}

	for _, tc := range []struct {
		name      string
		decisions []Decision
		want      bool
	}{
		{"masking enabled and no decision reads its pii rule", []Decision{masking(true)}, true},
		{"masking disabled and no decision reads its pii rule", []Decision{masking(false)}, false},
		{"no masking plugin at all", []Decision{{Name: "plain", ModelRefs: []ModelRef{{Model: "m"}}}}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &RouterConfig{
				IntelligentRouting: IntelligentRouting{
					Signals:   Signals{PIIRules: []PIIRule{{Name: "personal"}}},
					Decisions: tc.decisions,
				},
			}
			// The default auto model names make the default recipe reachable,
			// so the walk has somewhere to look.
			if !cfg.IsRecipeReachableForRouting(DefaultRecipeName) {
				t.Fatal("default recipe is unreachable; the walk would be vacuous")
			}
			rules := cfg.ApplicationRules()
			if len(rules) != 1 {
				t.Fatalf("ApplicationRules() = %+v, want the pii rule", rules)
			}
			if got := cfg.AsksApplicationRule(rules[0]); got != tc.want {
				t.Fatalf("AsksApplicationRule(personal) = %v, want %v", got, tc.want)
			}
		})
	}
}
