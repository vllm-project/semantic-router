package config

import "testing"

// Masking consumes the PII classifier without declaring any pii rule, so the
// signal-type walk alone reported the classifier unused: the mapping stayed
// unloaded, IsPIIEnabled stayed false, and because masking fails closed every
// request on a masking-only recipe answered 503. Found by the first cluster
// run of the masking e2e profile (#3806 review).
func TestMaskingPluginAloneRequiresPIIClassifier(t *testing.T) {
	maskingDecision := Decision{
		Name:      "mask_only",
		ModelRefs: []ModelRef{{Model: "m"}},
		Plugins: []DecisionPlugin{{
			Type:          DecisionPluginMasking,
			Configuration: MustStructuredPayload(map[string]interface{}{"enabled": true}),
		}},
	}
	disabledMasking := Decision{
		Name:      "mask_off",
		ModelRefs: []ModelRef{{Model: "m"}},
		Plugins: []DecisionPlugin{{
			Type:          DecisionPluginMasking,
			Configuration: MustStructuredPayload(map[string]interface{}{"enabled": false}),
		}},
	}

	for _, tc := range []struct {
		name      string
		decisions []Decision
		want      bool
	}{
		{"masking enabled and no pii rule", []Decision{maskingDecision}, true},
		{"masking disabled and no pii rule", []Decision{disabledMasking}, false},
		{"no masking plugin at all", []Decision{{Name: "plain", ModelRefs: []ModelRef{{Model: "m"}}}}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := &RouterConfig{
				IntelligentRouting: IntelligentRouting{Decisions: tc.decisions},
			}
			// The default auto model names make the default recipe reachable,
			// so the walk has somewhere to look.
			if !cfg.IsRecipeReachableForRouting(DefaultRecipeName) {
				t.Fatal("default recipe is unreachable; the walk would be vacuous")
			}
			if got := cfg.UsesPIIClassifierInReachableRouting(); got != tc.want {
				t.Fatalf("UsesPIIClassifierInReachableRouting() = %v, want %v", got, tc.want)
			}
		})
	}
}
