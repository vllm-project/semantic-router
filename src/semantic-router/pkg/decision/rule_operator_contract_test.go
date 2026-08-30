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
				assertOperatorReachesItsOwnBranch(t, spelling, tc.conditions, tc.wantMatch)
			})
		}
	}
}

// assertOperatorReachesItsOwnBranch drives one operator spelling through both
// evaluators and checks the verdict they produce, including the flag the trace
// reports for itself.
func assertOperatorReachesItsOwnBranch(
	t *testing.T,
	operator string,
	conditions []config.RuleCondition,
	wantMatch bool,
) {
	t.Helper()
	engine := NewDecisionEngine(nil, nil, nil, []config.Decision{
		ruleDecision("d", 10, operator, conditions...),
	}, config.RoutingStrategyPriority)
	signals := &SignalMatches{KeywordRules: []string{"present"}}

	result, err := engine.EvaluateDecisionsWithSignals(signals)
	if err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals: %v", err)
	}
	if matched := result != nil; matched != wantMatch {
		t.Fatalf("evalNode, operator %q: matched=%v, want %v "+
			"(an operator that reaches the default branch is evaluated as OR)",
			operator, matched, wantMatch)
	}

	tracedResult, traces := engine.EvaluateDecisionsWithTrace(signals)
	if matched := tracedResult != nil; matched != wantMatch {
		t.Fatalf("evalNodeWithTrace, operator %q: matched=%v, want %v",
			operator, matched, wantMatch)
	}
	if len(traces) != 1 {
		t.Fatalf("expected one decision trace, got %d", len(traces))
	}
	if traces[0].Matched != wantMatch {
		t.Fatalf("trace for operator %q reports matched=%v but the decision is %v",
			operator, traces[0].Matched, wantMatch)
	}
}
