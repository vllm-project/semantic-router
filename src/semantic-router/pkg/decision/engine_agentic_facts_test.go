package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDecisionEngine_EvaluateDecisionsWithAgenticFacts(t *testing.T) {
	tests := []evaluateSignalsCase{
		{
			name: "reviewer role matches",
			decisions: []config.Decision{ruleDecision(
				"reviewer-route", 10, "AND",
				config.RuleCondition{Type: config.SignalTypeAgenticFacts, Name: "reviewer-role"},
			)},
			signals:          &SignalMatches{AgenticFactsRules: []string{"reviewer-role"}},
			expectedDecision: "reviewer-route",
		},
		{
			name: "reviewer role does not match",
			decisions: []config.Decision{ruleDecision(
				"reviewer-route", 10, "AND",
				config.RuleCondition{Type: config.SignalTypeAgenticFacts, Name: "reviewer-role"},
			)},
			signals: &SignalMatches{AgenticFactsRules: []string{"execution-phase"}},
		},
		{
			name: "agentic facts combined with domain",
			decisions: []config.Decision{ruleDecision(
				"reviewer-science", 10, "AND",
				config.RuleCondition{Type: config.SignalTypeAgenticFacts, Name: "reviewer-role"},
				config.RuleCondition{Type: config.SignalTypeDomain, Name: "science"},
			)},
			signals: &SignalMatches{
				AgenticFactsRules: []string{"reviewer-role"},
				DomainRules:       []string{"science"},
			},
			expectedDecision: "reviewer-science",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			engine := NewDecisionEngine(nil, nil, nil, tt.decisions, config.RoutingStrategyPriority)
			result, err := engine.EvaluateDecisionsWithSignals(tt.signals)
			assertDecisionResult(t, result, err, tt.expectedDecision)
		})
	}
}
