package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDecisionEngine_EvaluateDecisionsWithRandom(t *testing.T) {
	engine := NewDecisionEngine(nil, nil, nil, []config.Decision{
		{
			Name:     "random_route",
			Priority: 1,
			Rules: config.RuleNode{
				Type: config.SignalTypeRandom,
				Name: "random_digit",
			},
		},
	}, "priority")

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		RandomRules: []string{"random_digit"},
		SignalConfidences: map[string]float64{
			"random:random_digit": 1.0,
		},
	})
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals() error = %v", err)
	}
	if result == nil || result.Decision.Name != "random_route" {
		t.Fatalf("decision = %+v, want random_route", result)
	}
}
