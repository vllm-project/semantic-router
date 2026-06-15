package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestEvaluateDecisionsWithTrace_BasicLeaf(t *testing.T) {
	engine := NewDecisionEngine(
		[]config.KeywordRule{{Name: "code_help", Keywords: []string{"code"}}},
		nil,
		nil,
		[]config.Decision{
			{
				Name:     "code_route",
				Priority: 10,
				Rules: config.RuleNode{
					Type: "keyword",
					Name: "code_help",
				},
			},
		},
		"priority",
	)

	signals := &SignalMatches{
		KeywordRules: []string{"code_help"},
	}

	result, traces := engine.EvaluateDecisionsWithTrace(signals)
	if result == nil {
		t.Fatal("expected a decision result")
	}
	if result.Decision.Name != "code_route" {
		t.Errorf("expected code_route, got %s", result.Decision.Name)
	}
	if len(traces) != 1 {
		t.Fatalf("expected 1 trace, got %d", len(traces))
	}
	if !traces[0].Matched {
		t.Error("expected trace to show matched")
	}
	if traces[0].RootTrace == nil {
		t.Fatal("expected root trace node")
	}
	if traces[0].RootTrace.NodeType != "leaf" {
		t.Errorf("expected leaf node type, got %s", traces[0].RootTrace.NodeType)
	}
	if !traces[0].RootTrace.Matched {
		t.Error("expected root trace to show matched")
	}
}

func TestEvaluateDecisionsWithTrace_ANDWithOneFailing(t *testing.T) {
	engine := NewDecisionEngine(
		[]config.KeywordRule{
			{Name: "code_help", Keywords: []string{"code"}},
			{Name: "math_help", Keywords: []string{"math"}},
		},
		nil,
		nil,
		[]config.Decision{
			{
				Name:     "both_route",
				Priority: 10,
				Rules: config.RuleNode{
					Operator: "AND",
					Conditions: []config.RuleNode{
						{Type: "keyword", Name: "code_help"},
						{Type: "keyword", Name: "math_help"},
					},
				},
			},
		},
		"priority",
	)

	signals := &SignalMatches{
		KeywordRules: []string{"code_help"},
	}

	result, traces := engine.EvaluateDecisionsWithTrace(signals)
	if result != nil {
		t.Error("expected no match when one AND branch fails")
	}
	if len(traces) != 1 {
		t.Fatalf("expected 1 trace, got %d", len(traces))
	}
	if traces[0].Matched {
		t.Error("trace should show not matched")
	}
	root := traces[0].RootTrace
	if root.NodeType != "AND" {
		t.Errorf("expected AND node, got %s", root.NodeType)
	}
	if len(root.Children) != 2 {
		t.Fatalf("expected 2 children, got %d", len(root.Children))
	}
	if !root.Children[0].Matched {
		t.Error("first child (code_help) should match")
	}
	if root.Children[1].Matched {
		t.Error("second child (math_help) should not match")
	}
}

func TestEvaluateDecisionsWithTrace_MultipleDecisions(t *testing.T) {
	engine := NewDecisionEngine(
		[]config.KeywordRule{
			{Name: "code_help", Keywords: []string{"code"}},
			{Name: "math_help", Keywords: []string{"math"}},
		},
		nil,
		nil,
		[]config.Decision{
			{
				Name:     "code_route",
				Priority: 10,
				Rules: config.RuleNode{
					Type: "keyword",
					Name: "code_help",
				},
			},
			{
				Name:     "math_route",
				Priority: 5,
				Rules: config.RuleNode{
					Type: "keyword",
					Name: "math_help",
				},
			},
		},
		"priority",
	)

	signals := &SignalMatches{
		KeywordRules: []string{"code_help"},
	}

	result, traces := engine.EvaluateDecisionsWithTrace(signals)
	if result == nil {
		t.Fatal("expected a result")
	}
	if result.Decision.Name != "code_route" {
		t.Errorf("expected code_route, got %s", result.Decision.Name)
	}
	if len(traces) != 2 {
		t.Fatalf("expected 2 traces (one per decision), got %d", len(traces))
	}
	if !traces[0].Matched {
		t.Error("code_route trace should be matched")
	}
	if traces[1].Matched {
		t.Error("math_route trace should not be matched")
	}
}

func TestFormatTrace(t *testing.T) {
	trace := DecisionTrace{
		DecisionName: "test_decision",
		Matched:      true,
		Confidence:   0.85,
		RootTrace: &TraceNode{
			NodeType:   "AND",
			Matched:    true,
			Confidence: 0.85,
			Children: []*TraceNode{
				{
					NodeType:   "leaf",
					SignalType: "keyword",
					SignalName: "code_help",
					Matched:    true,
					Confidence: 0.9,
				},
				{
					NodeType:   "leaf",
					SignalType: "domain",
					SignalName: "science",
					Matched:    true,
					Confidence: 0.8,
				},
			},
		},
	}
	output := FormatTrace(trace)
	if output == "" {
		t.Error("expected non-empty trace output")
	}
	if len(output) < 50 {
		t.Errorf("expected substantial trace output, got: %s", output)
	}
}
