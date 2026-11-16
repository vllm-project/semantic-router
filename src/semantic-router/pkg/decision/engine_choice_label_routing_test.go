package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A broad choice question, such as the action a request asks for, narrows a
// domain route without taking requests from the domain routes beside it.
func TestDecisionEngineKeepsDomainRoutesAheadOfBroadChoiceLabels(t *testing.T) {
	action := func(label string) config.RuleNode {
		return config.RuleNode{Type: config.SignalTypeSystemOne, Name: "action", Label: label}
	}
	domain := func(name string) config.RuleNode {
		return config.RuleNode{Type: config.SignalTypeDomain, Name: name}
	}
	decisions := []config.Decision{
		{Name: "math_decision", Priority: 10, Rules: config.RuleCombination{Operator: "OR", Conditions: []config.RuleNode{domain("math")}}},
		{Name: "biology_decision", Priority: 10, Rules: config.RuleCombination{Operator: "OR", Conditions: []config.RuleNode{domain("biology")}}},
		{Name: "explain_action", Priority: 30, Rules: config.RuleCombination{Operator: "AND", Conditions: []config.RuleNode{domain("computer science"), action("explain")}}},
		{Name: "fix_action", Priority: 30, Rules: config.RuleCombination{Operator: "AND", Conditions: []config.RuleNode{domain("computer science"), action("fix")}}},
		{Name: "generate_action", Priority: 30, Rules: config.RuleCombination{Operator: "AND", Conditions: []config.RuleNode{domain("computer science"), action("generate")}}},
	}

	engine := NewDecisionEngine(decisions, config.RoutingStrategyPriority)
	tests := []struct {
		name   string
		domain string
		action string
		want   string
	}{
		{"math question stays on math route even when the action is explain", "math", "explain", "math_decision"},
		{"biology question stays on biology route even when the action is explain", "biology", "explain", "biology_decision"},
		{"coding explain request reaches explain action route", "computer science", "explain", "explain_action"},
		{"coding fix request reaches fix action route", "computer science", "fix", "fix_action"},
		{"coding generate request reaches generate action route", "computer science", "generate", "generate_action"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
				DomainRules:    []string{tt.domain},
				SystemOneRules: []string{"action:" + tt.action},
			})
			if err != nil {
				t.Fatalf("EvaluateDecisionsWithSignals() error = %v", err)
			}
			if result == nil || result.Decision == nil || result.Decision.Name != tt.want {
				t.Fatalf("winner = %#v, want %s", result, tt.want)
			}
		})
	}
}
