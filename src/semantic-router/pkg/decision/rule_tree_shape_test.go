package decision

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestEvalNode_AgreesWithNormalizedOperatorSet is the runtime half of the
// rule-tree shape corpus in pkg/config/validator_rule_operator_test.go. Every
// tree that config.NormalizeRuleOperator accepts must evaluate through a named
// AND/OR/NOT branch to the truth-table result, and every tree it rejects is
// one the evaluator could only misinterpret (widen to OR, or negate a
// multi-child NOT into a permanent non-match), which is why validation refuses
// it before the engine ever sees it.
func TestEvalNode_AgreesWithNormalizedOperatorSet(t *testing.T) {
	kw := func(name string) config.RuleNode { return config.RuleNode{Type: config.SignalTypeKeyword, Name: name} }
	signals := &SignalMatches{KeywordRules: []string{"a", "b"}} // a=true, b=true, c=false

	cases := []struct {
		name     string
		rules    config.RuleNode
		accepted bool
		want     evaluationState // only meaningful when accepted
	}{
		{name: "omitted operator means AND", rules: config.RuleNode{Conditions: []config.RuleNode{kw("a"), kw("c")}}, accepted: true, want: evaluationFalse},
		{name: "lowercase and", rules: config.RuleNode{Operator: "and", Conditions: []config.RuleNode{kw("a"), kw("b")}}, accepted: true, want: evaluationTrue},
		{name: "OR", rules: config.RuleNode{Operator: "OR", Conditions: []config.RuleNode{kw("c"), kw("b")}}, accepted: true, want: evaluationTrue},
		{name: "padded not", rules: config.RuleNode{Operator: " not ", Conditions: []config.RuleNode{kw("c")}}, accepted: true, want: evaluationTrue},
		{name: "NOT of matched leaf", rules: config.RuleNode{Operator: "NOT", Conditions: []config.RuleNode{kw("a")}}, accepted: true, want: evaluationFalse},
		{name: "NOR via nesting", rules: config.RuleNode{Operator: "NOT", Conditions: []config.RuleNode{
			{Operator: "OR", Conditions: []config.RuleNode{kw("c"), kw("a")}},
		}}, accepted: true, want: evaluationFalse},
		{name: "NAND via nesting", rules: config.RuleNode{Operator: "NOT", Conditions: []config.RuleNode{
			{Operator: "AND", Conditions: []config.RuleNode{kw("c"), kw("a")}},
		}}, accepted: true, want: evaluationTrue},
		{name: "root childless AND matches", rules: config.RuleNode{Operator: "AND"}, accepted: true, want: evaluationTrue},

		{name: "XOR would widen to OR", rules: config.RuleNode{Operator: "XOR", Conditions: []config.RuleNode{kw("a"), kw("c")}}},
		{name: "multi-child NOT never matches", rules: config.RuleNode{Operator: "NOT", Conditions: []config.RuleNode{kw("c"), kw("c")}}},
		{name: "childless NOT never matches", rules: config.RuleNode{Operator: "NOT"}},
		{name: "root childless OR never matches", rules: config.RuleNode{Operator: "OR"}},
		{name: "nested childless OR never matches", rules: config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{kw("a"), {Operator: "OR"}}}},
		{name: "name without type never matches", rules: config.RuleNode{Operator: "AND", Conditions: []config.RuleNode{{Name: "a"}}}},
	}

	engine := NewDecisionEngine(nil, nil, nil, nil, config.RoutingStrategyPriority)
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			node := tc.rules
			err := config.NormalizeRuleOperator(&node)
			if !tc.accepted {
				if err == nil {
					t.Fatalf("expected config validation to reject this tree")
				}
				return
			}
			if err != nil {
				t.Fatalf("expected config validation to accept this tree, got: %v", err)
			}
			evaluation, _ := engine.evalNode(node, signals, false, false)
			if evaluation.state != tc.want {
				t.Fatalf("state = %s, want %s", evaluation.state, tc.want)
			}
		})
	}
}

// TestEvalNode_OperatorConstantsCoverEveryBranch pins the evaluator to the
// operator set exported by pkg/config, so adding an operator in one place
// without the other fails here instead of silently falling into a default
// branch.
func TestEvalNode_OperatorConstantsCoverEveryBranch(t *testing.T) {
	kw := func(name string) config.RuleNode { return config.RuleNode{Type: config.SignalTypeKeyword, Name: name} }
	signals := &SignalMatches{KeywordRules: []string{"a"}}
	engine := NewDecisionEngine(nil, nil, nil, nil, config.RoutingStrategyPriority)

	for _, op := range config.RuleTreeOperators() {
		children := []config.RuleNode{kw("a"), kw("missing")}
		if op == config.RuleOperatorNot {
			children = children[:1]
		}
		node := config.RuleNode{Operator: op, Conditions: children}
		_, trace := engine.evalNode(node, signals, false, true)
		if trace == nil || trace.NodeType != op {
			t.Fatalf("operator %q: expected trace node type %q, got %+v", op, op, trace)
		}
	}
}
