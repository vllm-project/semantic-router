package config

import "testing"

func TestARuleRefusesWhenADecisionAboveTheOthersRefusesOnItsMatchAlone(t *testing.T) {
	refuse := []DecisionPlugin{{Type: DecisionPluginFastResponse, Configuration: MustStructuredPayload(FastResponsePluginConfig{Message: "Blocked."})}}
	attack := RuleNode{Type: SignalTypeJailbreak, Name: "attack"}
	harm := RuleNode{Type: SignalTypeSafety, Name: "harm"}
	domain := RuleNode{Type: SignalTypeDomain, Name: "law"}
	route := Decision{Name: "law", Priority: 100, Rules: RuleNode{Operator: RuleOperatorAnd, Conditions: []RuleNode{domain}}}
	catchAll := Decision{Name: "default", Priority: 2000}
	for name, tc := range map[string]struct {
		decisions []Decision
		strategy  RoutingStrategy
		want      bool
	}{
		"or": {decisions: []Decision{
			{Name: "guard", Priority: 1000, Plugins: refuse, Rules: RuleNode{Operator: RuleOperatorOr, Conditions: []RuleNode{attack, harm}}}, route, catchAll,
		}, want: true},
		"the only condition": {decisions: []Decision{
			{Name: "guard", Priority: 1000, Plugins: refuse, Rules: RuleNode{Operator: RuleOperatorAnd, Conditions: []RuleNode{attack}}}, route,
		}, want: true},
		"and another condition": {decisions: []Decision{
			{Name: "guard", Priority: 1000, Plugins: refuse, Rules: RuleNode{Operator: RuleOperatorAnd, Conditions: []RuleNode{attack, domain}}}, route,
		}, want: false},
		"not": {decisions: []Decision{
			{Name: "guard", Priority: 1000, Plugins: refuse, Rules: RuleNode{Operator: RuleOperatorNot, Conditions: []RuleNode{attack}}}, route,
		}, want: false},
		"a label of the rule": {decisions: []Decision{
			{Name: "guard", Priority: 1000, Plugins: refuse, Rules: RuleNode{Operator: RuleOperatorOr, Conditions: []RuleNode{{Type: SignalTypeJailbreak, Name: "attack", Label: "jailbreak"}}}}, route,
		}, want: false},
		"routes the attack": {decisions: []Decision{
			{Name: "guard", Priority: 1000, Rules: RuleNode{Operator: RuleOperatorOr, Conditions: []RuleNode{attack}}, ModelRefs: []ModelRef{{Model: "safe"}}}, route,
		}, want: false},
		"under another decision": {decisions: []Decision{
			{Name: "guard", Priority: 50, Plugins: refuse, Rules: RuleNode{Operator: RuleOperatorOr, Conditions: []RuleNode{attack}}}, route,
		}, want: false},
		"as high as another": {decisions: []Decision{
			{Name: "guard", Priority: 100, Plugins: refuse, Rules: RuleNode{Operator: RuleOperatorOr, Conditions: []RuleNode{attack}}}, route,
		}, want: false},
		"in a lower tier": {decisions: []Decision{
			{Name: "guard", Tier: 1, Priority: 1000, Plugins: refuse, Rules: RuleNode{Operator: RuleOperatorOr, Conditions: []RuleNode{attack}}}, route,
		}, want: false},
		"in the first tier": {decisions: []Decision{
			{Name: "guard", Tier: 1, Priority: 10, Plugins: refuse, Rules: RuleNode{Operator: RuleOperatorOr, Conditions: []RuleNode{attack}}},
			{Name: "law", Tier: 2, Priority: 100, Rules: route.Rules},
		}, want: true},
		"routing by confidence": {decisions: []Decision{
			{Name: "guard", Priority: 1000, Plugins: refuse, Rules: RuleNode{Operator: RuleOperatorOr, Conditions: []RuleNode{attack}}}, route,
		}, strategy: RoutingStrategyConfidence, want: false},
	} {
		cfg := &RouterConfig{}
		cfg.Decisions, cfg.Strategy = tc.decisions, tc.strategy
		if got := cfg.RefusedOnMatch(SignalTypeJailbreak, "attack"); got != tc.want {
			t.Errorf("%s: refused on a match = %v, want %v", name, got, tc.want)
		}
	}
}
