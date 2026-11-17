package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestSystemOneLeaves(t *testing.T) {
	signals := &SignalMatches{
		SystemOneRules: []string{"hard", "kind:math", "kind", "topics", "topics:billing"},
		SignalValues: map[string]float64{
			"systemone:hard": 0.81, "systemone:kind": 0.7, "systemone:kind:math": 0.7, "systemone:kind:code": 0.3,
			"systemone:topics": 0.9, "systemone:topics:billing": 0.9, "systemone:topics:shipping": 0.2,
		},
		SignalConfidences: map[string]float64{"systemone:hard": 0.81, "systemone:kind": 0.7},
		SignalErrors:      map[string]string{"systemone:late": "systemone_timeout"},
	}
	engine := NewDecisionEngine(nil, config.RoutingStrategyPriority)
	tests := []struct {
		name string
		node config.RuleNode
		want evaluationState
	}{
		{"noul match", config.RuleNode{Type: "systemone", Name: "hard"}, evaluationTrue},
		{"choice rule match", config.RuleNode{Type: "systemone", Name: "kind"}, evaluationTrue},
		{"choice label match", config.RuleNode{Type: "systemone", Name: "kind", Label: "math"}, evaluationTrue},
		{"choice other label", config.RuleNode{Type: "systemone", Name: "kind", Label: "code"}, evaluationFalse},
		{"condition predicate on option", config.RuleNode{Type: "systemone", Name: "kind", Label: "code", Predicate: &config.NumericPredicate{GTE: float64Ptr(0.25)}}, evaluationTrue},
		{"set rule match", config.RuleNode{Type: "systemone", Name: "topics"}, evaluationTrue},
		{"set label match", config.RuleNode{Type: "systemone", Name: "topics", Label: "billing"}, evaluationTrue},
		{"set other label", config.RuleNode{Type: "systemone", Name: "topics", Label: "shipping"}, evaluationFalse},
		{"unknown by default no match", config.RuleNode{Type: "systemone", Name: "late"}, evaluationFalse},
		{"unknown with on_error match", config.RuleNode{Type: "systemone", Name: "late", OnError: "match"}, evaluationTrue},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			evaluation, _ := engine.evalNode(test.node, signals, "", false)
			if evaluation.state != test.want {
				t.Fatalf("state = %v, want %v", evaluation.state, test.want)
			}
		})
	}
	unknown, _ := engine.evalNode(config.RuleNode{Type: "systemone", Name: "late"}, signals, config.RuleOnUnknownNoMatch, false)
	if unknown.state != evaluationUnknown {
		t.Fatalf("a failed systemone answer under on_unknown must be unknown, got %v", unknown.state)
	}
	matched, _ := engine.evalNode(config.RuleNode{Type: "systemone", Name: "kind", Label: "math"}, signals, "", false)
	if matched.confidence != 0.7 || !matched.scored {
		t.Fatalf("choice leaves report the option probability: %+v", matched)
	}
}
