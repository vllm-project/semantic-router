package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// A decision that reads a response-stage signal must not be selected while the
// request is still being routed: the model has not answered, so the signal was
// never evaluated. Composed with a request signal it would otherwise win on its
// own priority off the request-stage half of the rule alone.
func TestDecisionEngine_SkipsNestedResponseStageDecisions(t *testing.T) {
	engine := NewDecisionEngine(
		nil, nil, nil,
		[]config.Decision{
			{
				Name:     "escalate-on-unsafe-output",
				Priority: 100,
				Rules: config.RuleCombination{
					Operator: "OR",
					Conditions: []config.RuleCondition{
						{Type: config.SignalTypeDomain, Name: "business"},
						{Type: config.SignalTypeResponseJailbreak, Name: "unsafe_completion"},
					},
				},
			},
			{
				Name:     "business",
				Priority: 10,
				Rules: config.RuleCombination{
					Type: config.SignalTypeDomain,
					Name: "business",
				},
			},
		},
		"priority",
	)

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		DomainRules: []string{"business"},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil || result.Decision.Name != "business" {
		t.Fatalf("a decision is response-stage wherever the response signal sits in its tree")
	}
}
