package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDecisionEngineKeepsDomainRoutesAheadOfBroadExplainAction(t *testing.T) {
	decisions := []config.Decision{
		{
			Name:     "math_decision",
			Priority: 10,
			Rules: config.RuleCombination{
				Operator: "OR",
				Conditions: []config.RuleNode{{
					Type: config.SignalTypeDomain,
					Name: "math",
				}},
			},
		},
		{
			Name:     "biology_decision",
			Priority: 10,
			Rules: config.RuleCombination{
				Operator: "OR",
				Conditions: []config.RuleNode{{
					Type: config.SignalTypeDomain,
					Name: "biology",
				}},
			},
		},
		{
			Name:     "explain_action",
			Priority: 30,
			Rules: config.RuleCombination{
				Operator: "AND",
				Conditions: []config.RuleNode{
					{Type: config.SignalTypeDomain, Name: "computer science"},
					{Type: config.SignalTypeAction, Name: config.ActionExplain},
				},
			},
		},
		{
			Name:     "fix_action",
			Priority: 30,
			Rules: config.RuleCombination{
				Operator: "AND",
				Conditions: []config.RuleNode{
					{Type: config.SignalTypeDomain, Name: "computer science"},
					{Type: config.SignalTypeAction, Name: config.ActionFix},
				},
			},
		},
		{
			Name:     "generate_action",
			Priority: 30,
			Rules: config.RuleCombination{
				Operator: "AND",
				Conditions: []config.RuleNode{
					{Type: config.SignalTypeDomain, Name: "computer science"},
					{Type: config.SignalTypeAction, Name: config.ActionGenerate},
				},
			},
		},
	}

	engine := NewDecisionEngine(nil, nil, nil, decisions, config.RoutingStrategyPriority)
	tests := []struct {
		name    string
		signals *SignalMatches
		want    string
	}{
		{
			name: "math question stays on math route even when action says explain",
			signals: &SignalMatches{
				DomainRules: []string{"math"},
				ActionRules: []string{config.ActionExplain},
			},
			want: "math_decision",
		},
		{
			name: "biology question stays on biology route even when action says explain",
			signals: &SignalMatches{
				DomainRules: []string{"biology"},
				ActionRules: []string{config.ActionExplain},
			},
			want: "biology_decision",
		},
		{
			name: "coding explain request reaches explain action route",
			signals: &SignalMatches{
				DomainRules: []string{"computer science"},
				ActionRules: []string{config.ActionExplain},
			},
			want: "explain_action",
		},
		{
			name: "coding fix request reaches fix action route",
			signals: &SignalMatches{
				DomainRules: []string{"computer science"},
				ActionRules: []string{config.ActionFix},
			},
			want: "fix_action",
		},
		{
			name: "coding generate request reaches generate action route",
			signals: &SignalMatches{
				DomainRules: []string{"computer science"},
				ActionRules: []string{config.ActionGenerate},
			},
			want: "generate_action",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := engine.EvaluateDecisionsWithSignals(tt.signals)
			if err != nil {
				t.Fatalf("EvaluateDecisionsWithSignals() error = %v", err)
			}
			if result == nil || result.Decision == nil || result.Decision.Name != tt.want {
				t.Fatalf("winner = %#v, want %s", result, tt.want)
			}
		})
	}
}
