package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDecisionEngine_SelectBestDecisionPrefersLowerTier(t *testing.T) {
	engine := NewDecisionEngine(
		nil,
		nil,
		nil,
		[]config.Decision{
			testDecision("jailbreak-block", "jailbreak", "detector", 10, 1),
			testDecision("math-route", "domain", "math", 200, 2),
		},
		"priority",
	)

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		DomainRules:       []string{"math"},
		JailbreakRules:    []string{"detector"},
		SignalConfidences: map[string]float64{"domain:math": 0.99, "jailbreak:detector": 0.55},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("expected a matched decision")
	}
	if result.Decision.Name != "jailbreak-block" {
		t.Fatalf("expected lower-tier decision to win, got %s", result.Decision.Name)
	}
}

func TestDecisionEngine_SelectBestDecisionUsesConfidenceWithinTier(t *testing.T) {
	engine := NewDecisionEngine(
		nil,
		nil,
		nil,
		[]config.Decision{
			testDecision("math-route", "domain", "math", 200, 2),
			testDecision("science-route", "domain", "science", 100, 2),
		},
		"priority",
	)

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		DomainRules: []string{"math", "science"},
		SignalConfidences: map[string]float64{
			"domain:math":    0.52,
			"domain:science": 0.89,
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("expected a matched decision")
	}
	if result.Decision.Name != "science-route" {
		t.Fatalf("expected higher-confidence same-tier decision to win, got %s", result.Decision.Name)
	}
}

func TestDecisionEngine_SelectBestDecisionFallsBackToPriorityOnTierConfidenceTie(t *testing.T) {
	engine := NewDecisionEngine(
		nil,
		nil,
		nil,
		[]config.Decision{
			testDecision("math-route", "domain", "math", 200, 2),
			testDecision("science-route", "domain", "science", 100, 2),
		},
		"priority",
	)

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		DomainRules: []string{"math", "science"},
		SignalConfidences: map[string]float64{
			"domain:math":    0.89,
			"domain:science": 0.89,
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("expected a matched decision")
	}
	if result.Decision.Name != "math-route" {
		t.Fatalf("expected higher-priority tie-breaker to win, got %s", result.Decision.Name)
	}
}

func TestDecisionEngine_SelectBestDecisionKeepsLegacyPriorityWithoutTiers(t *testing.T) {
	engine := NewDecisionEngine(
		nil,
		nil,
		nil,
		[]config.Decision{
			testDecision("math-route", "domain", "math", 200, 0),
			testDecision("science-route", "domain", "science", 100, 0),
		},
		"priority",
	)

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		DomainRules: []string{"math", "science"},
		SignalConfidences: map[string]float64{
			"domain:math":    0.52,
			"domain:science": 0.89,
		},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result == nil {
		t.Fatal("expected a matched decision")
	}
	if result.Decision.Name != "math-route" {
		t.Fatalf("expected legacy priority selection without tiers, got %s", result.Decision.Name)
	}
}

func testDecision(name string, signalType string, signalName string, priority int, tier int) config.Decision {
	return config.Decision{
		Name:     name,
		Priority: priority,
		Tier:     tier,
		Rules: config.RuleCombination{
			Operator: "OR",
			Conditions: []config.RuleCondition{
				{Type: signalType, Name: signalName},
			},
		},
	}
}
