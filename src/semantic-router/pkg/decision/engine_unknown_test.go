package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestUnknownTruthTable(t *testing.T) {
	unknown := config.RuleNode{
		Type:      config.SignalTypeClassifier,
		Name:      "risk",
		Label:     "RISKY",
		Predicate: &config.NumericPredicate{GTE: float64Ptr(0.5)},
	}
	truthy := config.RuleNode{Type: config.SignalTypeKeyword, Name: "present"}
	falsy := config.RuleNode{Type: config.SignalTypeKeyword, Name: "missing"}
	tests := []struct {
		name string
		rule config.RuleNode
		want evaluationState
	}{
		{"not", config.RuleNode{Operator: "NOT", Conditions: []config.RuleNode{unknown}}, evaluationUnknown},
		{"false and unknown", config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{falsy, unknown}}, evaluationFalse},
		{"true and unknown", config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{truthy, unknown}}, evaluationUnknown},
		{"true or unknown", config.RuleNode{Operator: "OR", Conditions: []config.RuleNode{truthy, unknown}}, evaluationTrue},
		{"false or unknown", config.RuleNode{Operator: "OR", Conditions: []config.RuleNode{falsy, unknown}}, evaluationUnknown},
	}
	signals := &SignalMatches{
		KeywordRules: []string{"present"},
		SignalErrors: map[string]string{"classifier:risk": "timeout"},
	}
	engine := NewDecisionEngine(nil, nil, nil, nil, config.RoutingStrategyPriority)
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			evaluation, _ := engine.evalNode(test.rule, signals, false, false)
			if evaluation.state != test.want {
				t.Fatalf("state = %v, want %v", evaluation.state, test.want)
			}
			traced, trace := engine.evalNode(test.rule, signals, false, true)
			if traced.state != test.want || trace == nil {
				t.Fatalf("traced state = %v (trace %v), want %v", traced.state, trace, test.want)
			}
		})
	}
}

func TestUnknownOrTrueMatchesResolvedBranch(t *testing.T) {
	rule := config.RuleNode{Operator: "OR", Conditions: []config.RuleNode{
		{Type: config.SignalTypeJailbreak, Name: "guard"},
		{Type: config.SignalTypeKeyword, Name: "present"},
	}}
	signals := &SignalMatches{
		KeywordRules:       []string{"present"},
		JailbreakRules:     []string{"guard"},
		SignalConfidences:  map[string]float64{"keyword:present": 0.6},
		SignalErrors:       map[string]string{"jailbreak:guard": "unavailable"},
		SignalErrorMatches: map[string]bool{"jailbreak:guard": true},
	}
	engine := NewDecisionEngine(nil, nil, nil, []config.Decision{{Name: "route", Rules: rule}}, config.RoutingStrategyPriority)
	result, err := engine.EvaluateDecisionsWithSignals(signals)
	if err != nil || result == nil {
		t.Fatalf("result = %#v, error = %v", result, err)
	}
	if len(result.MatchedRules) != 1 || result.MatchedRules[0] != "keyword:present" {
		t.Fatalf("matched rules = %v, want [keyword:present]", result.MatchedRules)
	}
	if result.Confidence != 0.6 {
		t.Fatalf("confidence = %v, want 0.6", result.Confidence)
	}
	traceResult, traces := engine.EvaluateDecisionsWithTrace(signals)
	if traceResult == nil || len(traces) != 1 {
		t.Fatalf("trace result = %#v, traces = %v", traceResult, traces)
	}
	if traceResult.Confidence != result.Confidence {
		t.Fatalf("trace confidence = %v, want %v", traceResult.Confidence, result.Confidence)
	}
}

func TestUnknownKeepsLegacyClassifierOnError(t *testing.T) {
	rule := config.RuleNode{
		Type:      config.SignalTypeClassifier,
		Name:      "risk",
		Label:     "RISKY",
		Predicate: &config.NumericPredicate{GTE: float64Ptr(0.5)},
		OnError:   "match",
	}
	engine := NewDecisionEngine(nil, nil, nil, []config.Decision{{Name: "route", Rules: rule}}, config.RoutingStrategyPriority)
	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		SignalErrors: map[string]string{"classifier:risk": "timeout"},
	})
	if err != nil || result == nil {
		t.Fatalf("result = %#v, error = %v", result, err)
	}
}

func TestUnknownKeepsLegacyPromptGuardResult(t *testing.T) {
	rule := config.RuleNode{Type: config.SignalTypeJailbreak, Name: "guard"}
	engine := NewDecisionEngine(nil, nil, nil, []config.Decision{{Name: "route", Rules: rule}}, config.RoutingStrategyPriority)
	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		JailbreakRules:     []string{"guard"},
		SignalErrors:       map[string]string{"jailbreak:guard": "unavailable"},
		SignalErrorMatches: map[string]bool{"jailbreak:guard": true},
	})
	if err != nil || result == nil {
		t.Fatalf("result = %#v, error = %v", result, err)
	}
}
