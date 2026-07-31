package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func float64Ptr(value float64) *float64 { return &value }

func TestScorePredicateUsesRawSignalValueWithoutBooleanMatch(t *testing.T) {
	engine := NewDecisionEngine(nil, nil, nil, []config.Decision{{
		Name: "bounded",
		Rules: config.RuleNode{
			Type: "embedding",
			Name: "risk",
			Predicate: &config.NumericPredicate{
				GTE: float64Ptr(0.2),
				LT:  float64Ptr(0.4),
			},
		},
	}}, "priority")

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		SignalValues: map[string]float64{"embedding:risk": 0.3},
	})
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals() error = %v", err)
	}
	if result == nil || result.Decision == nil || result.Decision.Name != "bounded" {
		t.Fatalf("result = %#v, want bounded decision", result)
	}
}

func TestSignalConfidencePreservesZero(t *testing.T) {
	if got := signalConfidence(map[string]float64{"embedding:risk": 0}, "embedding", "risk"); got != 0 {
		t.Fatalf("signalConfidence() = %v, want 0", got)
	}
}

func TestScorePredicateOnErrorMatch(t *testing.T) {
	engine := NewDecisionEngine(nil, nil, nil, []config.Decision{{
		Name: "fail-closed",
		Rules: config.RuleNode{
			Type:      "embedding",
			Name:      "risk",
			Predicate: &config.NumericPredicate{GTE: float64Ptr(0.8)},
			OnError:   "match",
		},
	}}, "priority")

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		SignalErrors: map[string]string{"embedding:risk": "backend unavailable"},
	})
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals() error = %v", err)
	}
	if result == nil || result.Decision.Name != "fail-closed" {
		t.Fatalf("result = %#v, want fail-closed decision", result)
	}
}
