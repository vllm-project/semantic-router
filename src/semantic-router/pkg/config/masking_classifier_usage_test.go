package config

import (
	"slices"
	"strings"
	"testing"
)

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

// A request that matches no decision masks with the recipe's default masking
// plugin, so its pii rule must exist and is asked even when every decision
// masks with another rule.
func TestDefaultMaskingPluginNamesARuleOfTheRecipe(t *testing.T) {
	doc := func(signal string) string {
		return `version: v0.5
providers:
  models:
    - name: model-a
      backend_refs: [{endpoint: "127.0.0.1:8000", provider: vllm}]
recipes:
  - name: default
    signals:
      pii:
        - {name: personal, predicate: {gte: 0.5}}
        - {name: contact, predicate: {gte: 0.5}}
    decisions:
      - name: mask
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: model-a}]
        plugins: [{type: masking, configuration: {enabled: true, signal: personal}}]
global:
  plugins: [{type: masking, configuration: {enabled: true, signal: ` + signal + `}}]
`
	}
	cfg, err := ParseYAMLBytes([]byte(doc("contact")))
	if err != nil {
		t.Fatalf("a default masking plugin on a pii rule of the recipe must load: %v", err)
	}
	if got := cfg.MaskingSignals(); !slices.Equal(got, []string{"personal", "contact"}) {
		t.Fatalf("MaskingSignals() = %v, want the decision's rule and the default plugins'", got)
	}
	_, err = ParseYAMLBytes([]byte(doc("missing")))
	if err == nil || !strings.Contains(err.Error(), `recipes[default].plugins masking: signal "missing" names no pii rule of this recipe`) {
		t.Fatalf("err = %v, want the default masking plugin's unknown rule refused", err)
	}
}
