package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestDecisionModelLeaves(t *testing.T) {
	signals := &SignalMatches{
		DecisionRules: []string{"hard", "kind:math"},
		SignalValues: map[string]float64{
			"decision:hard": 0.81, "decision:kind": 0.7, "decision:kind:math": 0.7, "decision:kind:code": 0.3,
		},
		SignalConfidences: map[string]float64{"decision:hard": 0.81, "decision:kind": 0.7},
		SignalErrors:      map[string]string{"decision:late": "decision_timeout"},
	}
	engine := NewDecisionEngine(nil, nil, nil, nil, config.RoutingStrategyPriority)
	tests := []struct {
		name string
		node config.RuleNode
		want evaluationState
	}{
		{"noul match", config.RuleNode{Type: "decision", Name: "hard"}, evaluationTrue},
		{"choice label match", config.RuleNode{Type: "decision", Name: "kind", Label: "math"}, evaluationTrue},
		{"choice other label", config.RuleNode{Type: "decision", Name: "kind", Label: "code"}, evaluationFalse},
		{"condition predicate on option", config.RuleNode{Type: "decision", Name: "kind", Label: "code", Predicate: &config.NumericPredicate{GTE: float64Ptr(0.25)}}, evaluationTrue},
		{"unknown by default no match", config.RuleNode{Type: "decision", Name: "late"}, evaluationFalse},
		{"unknown with on_error match", config.RuleNode{Type: "decision", Name: "late", OnError: "match"}, evaluationTrue},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			evaluation, _ := engine.evalNode(test.node, signals, "", false)
			if evaluation.state != test.want {
				t.Fatalf("state = %v, want %v", evaluation.state, test.want)
			}
		})
	}
	unknown, _ := engine.evalNode(config.RuleNode{Type: "decision", Name: "late"}, signals, config.RuleOnUnknownNoMatch, false)
	if unknown.state != evaluationUnknown {
		t.Fatalf("a failed decision answer under on_unknown must be unknown, got %v", unknown.state)
	}
	matched, _ := engine.evalNode(config.RuleNode{Type: "decision", Name: "kind", Label: "math"}, signals, "", false)
	if matched.confidence != 0.7 || !matched.scored {
		t.Fatalf("choice leaves report the option probability: %+v", matched)
	}
}
