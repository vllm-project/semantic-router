package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDecisionEngine_EmptyANDActsAsCatchAll(t *testing.T) {
	engine := NewDecisionEngine(
		nil,
		nil,
		nil,
		[]config.Decision{
			{
				Name:     "default-route",
				Priority: 10,
				Rules: config.RuleCombination{
					Operator:   "AND",
					Conditions: []config.RuleCondition{},
				},
			},
		},
		"priority",
	)

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("Expected result but got nil")
	}
	if result.Decision.Name != "default-route" {
		t.Fatalf("Expected default-route, got %s", result.Decision.Name)
	}
	if result.Confidence != 0.0 {
		t.Fatalf("Expected zero confidence for catch-all route, got %f", result.Confidence)
	}
}

func TestDecisionEngine_EmptyANDActsAsFallbackInConfidenceMode(t *testing.T) {
	engine := NewDecisionEngine(
		nil,
		nil,
		nil,
		[]config.Decision{
			{
				Name:     "specific-route",
				Priority: 200,
				Rules: config.RuleCombination{
					Operator: "AND",
					Conditions: []config.RuleCondition{
						{Type: "keyword", Name: "urgent"},
					},
				},
			},
			{
				Name:     "default-route",
				Priority: 100,
				Rules: config.RuleCombination{
					Operator:   "AND",
					Conditions: []config.RuleCondition{},
				},
			},
		},
		"confidence",
	)

	t.Run("signal-backed route outranks catch-all", func(t *testing.T) {
		result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
			KeywordRules: []string{"urgent"},
			SignalConfidences: map[string]float64{
				"keyword:urgent": 0.72,
			},
		})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if result == nil {
			t.Fatal("Expected result but got nil")
		}
		if result.Decision.Name != "specific-route" {
			t.Fatalf("Expected specific-route, got %s", result.Decision.Name)
		}
	})

	t.Run("catch-all still matches without signals", func(t *testing.T) {
		result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if result == nil {
			t.Fatal("Expected result but got nil")
		}
		if result.Decision.Name != "default-route" {
			t.Fatalf("Expected default-route, got %s", result.Decision.Name)
		}
		if result.Confidence != 0.0 {
			t.Fatalf("Expected zero confidence for catch-all route, got %f", result.Confidence)
		}
	})
}
