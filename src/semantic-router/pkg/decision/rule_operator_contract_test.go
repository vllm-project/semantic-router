package decision

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Config validation accepts exactly config.RuleTreeOperators and rejects
// everything else, which is only sound while every accepted operator reaches its
// own branch in both evaluators. evalNode and evalNodeWithTrace each carry their
// own copy of the operator switch, so an operator added to the set with a case in
// only one of them would fall through the other's default branch and be evaluated
// as OR — and a trace that disagrees with the decision actually taken is worse
// than no trace at all.
//
// Each case below has a different verdict for every operator, so a passing case
// proves the operator reached its own branch rather than the default one.
func TestRuleTreeOperatorsAgreeWithEvaluator(t *testing.T) {
	present := config.RuleCondition{Type: config.SignalTypeKeyword, Name: "present"}
	absent := config.RuleCondition{Type: config.SignalTypeKeyword, Name: "absent"}

	expected := map[string]struct {
		conditions []config.RuleCondition
		wantMatch  bool
	}{
		config.RuleOperatorAnd: {[]config.RuleCondition{present, absent}, false},
		config.RuleOperatorOr:  {[]config.RuleCondition{present, absent}, true},
		config.RuleOperatorNot: {[]config.RuleCondition{absent}, true},
	}

	if len(expected) != len(config.RuleTreeOperators()) {
		t.Fatalf("config.RuleTreeOperators has %d operators but this contract covers %d: "+
			"teach evalNode and this test about the new operator",
			len(config.RuleTreeOperators()), len(expected))
	}

	for _, operator := range config.RuleTreeOperators() {
		tc, ok := expected[operator]
		if !ok {
			t.Fatalf("operator %q is accepted by config validation but has no evaluator contract here", operator)
		}
		// Config validation matches the operator case-insensitively, so the
		// evaluator has to as well.
		for _, spelling := range []string{operator, strings.ToLower(operator)} {
			t.Run(spelling, func(t *testing.T) {
				engine := NewDecisionEngine(nil, nil, nil, []config.Decision{
					ruleDecision("d", 10, spelling, tc.conditions...),
				}, config.RoutingStrategyPriority)
				signals := &SignalMatches{KeywordRules: []string{"present"}}

				result, err := engine.EvaluateDecisionsWithSignals(signals)
				if err != nil {
					t.Fatalf("EvaluateDecisionsWithSignals: %v", err)
				}
				if matched := result != nil; matched != tc.wantMatch {
					t.Fatalf("evalNode, operator %q: matched=%v, want %v "+
						"(an operator that reaches the default branch is evaluated as OR)",
						spelling, matched, tc.wantMatch)
				}

				tracedResult, traces := engine.EvaluateDecisionsWithTrace(signals)
				if matched := tracedResult != nil; matched != tc.wantMatch {
					t.Fatalf("evalNodeWithTrace, operator %q: matched=%v, want %v",
						spelling, matched, tc.wantMatch)
				}
				if len(traces) != 1 {
					t.Fatalf("expected one decision trace, got %d", len(traces))
				}
				if traces[0].Matched != tc.wantMatch {
					t.Fatalf("trace for operator %q reports matched=%v but the decision is %v",
						spelling, traces[0].Matched, tc.wantMatch)
				}
			})
		}
	}
}
