package decision

import (
	"math"
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

func TestPredicateMatchesUseBooleanConfidenceForRanking(t *testing.T) {
	engine := NewDecisionEngine(nil, nil, nil, []config.Decision{
		{
			Name:     "strong-lower-bound-match",
			Priority: 10,
			Tier:     1,
			Rules: config.RuleNode{
				Type:      "structure",
				Name:      "small",
				Predicate: &config.NumericPredicate{LT: float64Ptr(0.2)},
			},
		},
		{
			Name:     "priority-wins-after-predicate-match",
			Priority: 20,
			Tier:     1,
			Rules: config.RuleNode{
				Type:      "structure",
				Name:      "large",
				Predicate: &config.NumericPredicate{GTE: float64Ptr(0.5)},
			},
		},
	}, "priority")

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		SignalValues: map[string]float64{
			"structure:small": 0.01,
			"structure:large": 0.5,
		},
	})
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals() error = %v", err)
	}
	if result == nil || result.Decision.Name != "priority-wins-after-predicate-match" {
		t.Fatalf("result = %#v, want priority-based winner", result)
	}
	if result.Confidence != 1 {
		t.Fatalf("confidence = %v, want boolean predicate confidence 1", result.Confidence)
	}
}

func TestScorePredicateCanMatchPublishedZeroValue(t *testing.T) {
	engine := NewDecisionEngine(nil, nil, nil, []config.Decision{{
		Name: "empty",
		Rules: config.RuleNode{
			Type:      "structure",
			Name:      "bytes",
			Predicate: &config.NumericPredicate{LTE: float64Ptr(0)},
		},
	}}, "priority")

	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		SignalValues: map[string]float64{"structure:bytes": 0},
	})
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals() error = %v", err)
	}
	if result == nil || result.Decision.Name != "empty" {
		t.Fatalf("result = %#v, want zero-value decision", result)
	}
}

func TestNumericPredicateRejectsNonFiniteSignalValue(t *testing.T) {
	predicate := &config.NumericPredicate{GTE: float64Ptr(0)}
	for _, value := range []float64{
		math.NaN(),
		math.Inf(1),
		math.Inf(-1),
	} {
		if numericPredicateMatches(value, predicate) {
			t.Fatalf("non-finite value %v matched predicate", value)
		}
	}
}

func TestORPreservesZeroConfidenceMatchedRule(t *testing.T) {
	engine := NewDecisionEngine(nil, nil, nil, []config.Decision{{
		Name: "zero-confidence",
		Rules: config.RuleNode{
			Operator: "OR",
			Conditions: []config.RuleNode{
				{Type: "keyword", Name: "zero"},
				{Type: "keyword", Name: "missing"},
			},
		},
	}}, "priority")
	signals := &SignalMatches{
		KeywordRules:      []string{"zero"},
		SignalConfidences: map[string]float64{"keyword:zero": 0},
	}

	result, traces := engine.EvaluateDecisionsWithTrace(signals)
	if result == nil || len(result.MatchedRules) != 1 ||
		result.MatchedRules[0] != "keyword:zero" {
		t.Fatalf("result = %#v", result)
	}
	if len(traces) != 1 || traces[0].RootTrace == nil ||
		!traces[0].RootTrace.Matched {
		t.Fatalf("traces = %#v", traces)
	}
}
