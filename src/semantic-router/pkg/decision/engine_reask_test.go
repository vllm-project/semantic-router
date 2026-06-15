package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDecisionEngine_EvaluateDecisionsWithReaskLikelyEscalation(t *testing.T) {
	engine := NewDecisionEngine(
		nil,
		nil,
		nil,
		[]config.Decision{{
			Name:     "repeat_question_escalation",
			Priority: 10,
			Rules: config.RuleCombination{
				Operator: "AND",
				Conditions: []config.RuleCondition{{
					Type: config.SignalTypeReask,
					Name: "likely_dissatisfied",
				}},
			},
		}},
		"priority",
	)

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		ReaskRules: []string{"likely_dissatisfied"},
	})
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals() error = %v", err)
	}
	if result == nil || result.Decision == nil {
		t.Fatal("expected matching decision")
	}
	if result.Decision.Name != "repeat_question_escalation" {
		t.Fatalf("decision name = %q, want repeat_question_escalation", result.Decision.Name)
	}
}

func TestDecisionEngine_EvaluateDecisionsWithReaskPersistentEscalation(t *testing.T) {
	engine := NewDecisionEngine(
		nil,
		nil,
		nil,
		[]config.Decision{{
			Name:     "persistent_repeat_escalation",
			Priority: 10,
			Rules: config.RuleCombination{
				Operator: "AND",
				Conditions: []config.RuleCondition{{
					Type: config.SignalTypeReask,
					Name: "persistently_dissatisfied",
				}},
			},
		}},
		"priority",
	)

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		ReaskRules: []string{"likely_dissatisfied"},
	})
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals() error = %v", err)
	}
	if result != nil {
		t.Fatalf("expected no match for likely_dissatisfied, got %+v", result)
	}

	result, err = engine.EvaluateDecisionsWithSignals(&SignalMatches{
		ReaskRules: []string{"persistently_dissatisfied"},
	})
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals() error = %v", err)
	}
	if result == nil || result.Decision == nil {
		t.Fatal("expected persistent reask decision to match")
	}
	if result.Decision.Name != "persistent_repeat_escalation" {
		t.Fatalf("decision name = %q, want persistent_repeat_escalation", result.Decision.Name)
	}
}
