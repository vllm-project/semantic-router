package decision

import (
	"errors"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestUnknownTruthTable(t *testing.T) {
	unknown := config.RuleNode{
		Type:      config.SignalTypeSystemOne,
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
		SignalErrors: map[string]string{"systemone:risk": "timeout"},
	}
	engine := NewDecisionEngine(nil, config.RoutingStrategyPriority)
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			evaluation, _ := engine.evalNode(test.rule, signals, config.RuleOnUnknownNoMatch, false)
			if evaluation.state != test.want {
				t.Fatalf("state = %v, want %v", evaluation.state, test.want)
			}
			traced, trace := engine.evalNode(test.rule, signals, config.RuleOnUnknownNoMatch, true)
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
	engine := NewDecisionEngine([]config.Decision{{Name: "route", Rules: rule}}, config.RoutingStrategyPriority)
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
	traceResult, traces, _, _ := engine.EvaluateDecisionsWithTraceAndDiagnostics(signals)
	if traceResult == nil || len(traces) != 1 {
		t.Fatalf("trace result = %#v, traces = %v", traceResult, traces)
	}
	if traceResult.Confidence != result.Confidence {
		t.Fatalf("trace confidence = %v, want %v", traceResult.Confidence, result.Confidence)
	}
}

func TestUnknownKeepsSystemOneOnError(t *testing.T) {
	rule := config.RuleNode{
		Type:      config.SignalTypeSystemOne,
		Name:      "risk",
		Label:     "RISKY",
		Predicate: &config.NumericPredicate{GTE: float64Ptr(0.5)},
		OnError:   "match",
	}
	engine := NewDecisionEngine([]config.Decision{{Name: "route", Rules: rule}}, config.RoutingStrategyPriority)
	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		SignalErrors: map[string]string{"systemone:risk": "timeout"},
	})
	if err != nil || result == nil {
		t.Fatalf("result = %#v, error = %v", result, err)
	}
}

func TestUnknownKeepsLegacyPromptGuardResult(t *testing.T) {
	rule := config.RuleNode{Type: config.SignalTypeJailbreak, Name: "guard"}
	engine := NewDecisionEngine([]config.Decision{{Name: "route", Rules: rule}}, config.RoutingStrategyPriority)
	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		JailbreakRules:     []string{"guard"},
		SignalErrors:       map[string]string{"jailbreak:guard": "unavailable"},
		SignalErrorMatches: map[string]bool{"jailbreak:guard": true},
	})
	if err != nil || result == nil {
		t.Fatalf("result = %#v, error = %v", result, err)
	}
}

func TestOnUnknownPolicies(t *testing.T) {
	rule := config.RuleNode{
		Type:      config.SignalTypeSystemOne,
		Name:      "risk",
		Label:     "RISKY",
		Predicate: &config.NumericPredicate{GTE: float64Ptr(0.5)},
		OnError:   "match",
	}
	signals := &SignalMatches{SignalErrors: map[string]string{"systemone:risk": "timeout"}}
	tests := []struct {
		name      string
		policy    config.UnknownPolicy
		wantMatch bool
		wantError bool
	}{
		{"legacy", "", true, false},
		{"no match", config.RuleOnUnknownNoMatch, false, false},
		{"match", config.RuleOnUnknownMatch, true, false},
		{"fail request", config.RuleOnUnknownFailRequest, false, true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			rule.OnUnknown = test.policy
			engine := NewDecisionEngine([]config.Decision{{Name: "route", Rules: rule}}, config.RoutingStrategyPriority)
			result, diagnostics, err := engine.EvaluateDecisionsWithDiagnostics(signals)
			if (err != nil) != test.wantError {
				t.Fatalf("error = %v, wantError %v", err, test.wantError)
			}
			if (result != nil) != test.wantMatch {
				t.Fatalf("result = %#v, wantMatch %v", result, test.wantMatch)
			}
			if diagnostics.AppliedUnknownPolicies["route"] != string(test.policy) {
				t.Fatalf("applied policies = %v, want %q", diagnostics.AppliedUnknownPolicies, test.policy)
			}
		})
	}
}

func TestUnknownLegacyTraceShowsResolvedLeaf(t *testing.T) {
	rule := config.RuleNode{
		Type:      config.SignalTypeSystemOne,
		Name:      "risk",
		Label:     "RISKY",
		Predicate: &config.NumericPredicate{GTE: float64Ptr(0.5)},
		OnError:   "match",
	}
	engine := NewDecisionEngine([]config.Decision{{Name: "route", Rules: rule}}, config.RoutingStrategyPriority)
	result, traces, diagnostics, err := engine.EvaluateDecisionsWithTraceAndDiagnostics(&SignalMatches{
		SignalErrors: map[string]string{"systemone:risk": "timeout"},
	})
	if err != nil || result == nil {
		t.Fatalf("result = %#v, error = %v", result, err)
	}
	if len(traces) != 1 || traces[0].State != "true" || !traces[0].Matched || traces[0].OnUnknown != "" {
		t.Fatalf("traces = %#v", traces)
	}
	if traces[0].RootTrace == nil || !traces[0].RootTrace.Matched {
		t.Fatalf("root trace = %#v", traces[0].RootTrace)
	}
	if len(diagnostics.AppliedUnknownPolicies) != 0 {
		t.Fatalf("diagnostics = %#v", diagnostics)
	}
}

func TestFailRequestEvaluatesAllDecisions(t *testing.T) {
	unresolved := config.RuleNode{
		Type:      config.SignalTypeSystemOne,
		Name:      "risk",
		Label:     "RISKY",
		Predicate: &config.NumericPredicate{GTE: float64Ptr(0.5)},
		OnUnknown: config.RuleOnUnknownFailRequest,
	}
	matching := config.RuleNode{Type: config.SignalTypeKeyword, Name: "present"}
	engine := NewDecisionEngine([]config.Decision{
		{Name: "guarded", Rules: unresolved},
		{Name: "route", Rules: matching},
	}, config.RoutingStrategyPriority)

	result, traces, _, err := engine.EvaluateDecisionsWithTraceAndDiagnostics(&SignalMatches{
		KeywordRules: []string{"present"},
		SignalErrors: map[string]string{"systemone:risk": "timeout"},
	})
	if !errors.Is(err, ErrDecisionUnresolved) || result != nil {
		t.Fatalf("result = %#v, error = %v", result, err)
	}
	if len(traces) != 2 || !traces[1].Matched {
		t.Fatalf("traces = %#v", traces)
	}
}

func TestFailRequestOverridesHigherPriorityMatch(t *testing.T) {
	guarded := config.Decision{Name: "guarded", Priority: 1, Rules: config.RuleNode{
		Type:      config.SignalTypeSystemOne,
		Name:      "risk",
		Label:     "RISKY",
		Predicate: &config.NumericPredicate{GTE: float64Ptr(0.5)},
		OnUnknown: config.RuleOnUnknownFailRequest,
	}}
	good := config.Decision{Name: "good", Priority: 100, Rules: config.RuleNode{
		Type: config.SignalTypeKeyword, Name: "present",
	}}
	signals := &SignalMatches{
		KeywordRules: []string{"present"},
		SignalErrors: map[string]string{"systemone:risk": "timeout"},
	}
	for name, decisions := range map[string][]config.Decision{
		"guarded first": {guarded, good},
		"good first":    {good, guarded},
	} {
		t.Run(name, func(t *testing.T) {
			engine := NewDecisionEngine(decisions, config.RoutingStrategyPriority)
			result, err := engine.EvaluateDecisionsWithSignals(signals)
			if !errors.Is(err, ErrDecisionUnresolved) || result != nil {
				t.Fatalf("result = %#v, error = %v", result, err)
			}
			var unresolved *DecisionUnresolvedError
			if !errors.As(err, &unresolved) || unresolved.Decision != "guarded" {
				t.Fatalf("error = %#v", err)
			}
		})
	}
}

func TestDecisionUnresolvedErrorDescribesUnknownEvidence(t *testing.T) {
	for _, reason := range []string{"user_feedback_uncertain", "user_feedback_evaluation_failed"} {
		t.Run(reason, func(t *testing.T) {
			engine := NewDecisionEngine([]config.Decision{{
				Name: "guarded",
				Rules: config.RuleNode{
					Type:      config.SignalTypeUserFeedback,
					Name:      "wrong_answer",
					OnUnknown: config.RuleOnUnknownFailRequest,
				},
			}}, config.RoutingStrategyPriority)

			result, traces, _, err := engine.EvaluateDecisionsWithTraceAndDiagnostics(&SignalMatches{
				SignalErrors: map[string]string{"user_feedback:wrong_answer": reason},
			})
			if result != nil || !errors.Is(err, ErrDecisionUnresolved) {
				t.Fatalf("result = %#v, error = %v", result, err)
			}
			for _, text := range []string{`decision "guarded"`, "unknown or unavailable", "Inspect the signal details", "rules.on_unknown"} {
				if !strings.Contains(err.Error(), text) {
					t.Fatalf("error %q does not include %q", err.Error(), text)
				}
			}
			if strings.Contains(err.Error(), "evaluator failed") || strings.Contains(err.Error(), "Fix the signal backend") {
				t.Fatalf("unknown evidence was attributed to a backend failure: %v", err)
			}
			if len(traces) != 1 || traces[0].State != "unknown" || traces[0].OnUnknown != string(config.RuleOnUnknownFailRequest) || traces[0].RootTrace == nil || traces[0].RootTrace.SignalError != reason {
				t.Fatalf("unknown state, policy or original reason changed: %#v", traces)
			}
		})
	}
}

func TestUnknownTraceRecordsErrorAndPolicy(t *testing.T) {
	rule := config.RuleNode{
		Type:      config.SignalTypeSystemOne,
		Name:      "risk",
		Label:     "RISKY",
		Predicate: &config.NumericPredicate{GTE: float64Ptr(0.5)},
		OnUnknown: config.RuleOnUnknownNoMatch,
	}
	engine := NewDecisionEngine([]config.Decision{{Name: "route", Rules: rule}}, config.RoutingStrategyPriority)
	_, traces, diagnostics, err := engine.EvaluateDecisionsWithTraceAndDiagnostics(&SignalMatches{
		SignalErrors: map[string]string{"systemone:risk": "timeout"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(traces) != 1 || traces[0].State != "unknown" || traces[0].OnUnknown != string(config.RuleOnUnknownNoMatch) {
		t.Fatalf("traces = %#v", traces)
	}
	if traces[0].RootTrace == nil || traces[0].RootTrace.SignalError != "timeout" {
		t.Fatalf("root trace = %#v", traces[0].RootTrace)
	}
	if diagnostics.AppliedUnknownPolicies["route"] != string(config.RuleOnUnknownNoMatch) {
		t.Fatalf("diagnostics = %#v", diagnostics)
	}
}

func TestOnErrorResolvedBranchNeverOutranksRealMatch(t *testing.T) {
	failed := config.RuleNode{
		Type:      config.SignalTypeSystemOne,
		Name:      "risk",
		Label:     "RISKY",
		Predicate: &config.NumericPredicate{GTE: float64Ptr(0.5)},
	}
	failedMatch := failed
	failedMatch.OnError = "match"
	present := config.RuleNode{Type: config.SignalTypeKeyword, Name: "present"}
	other := config.RuleNode{Type: config.SignalTypeKeyword, Name: "other"}
	signals := &SignalMatches{
		KeywordRules:      []string{"present", "other"},
		SignalConfidences: map[string]float64{"keyword:present": 0.6, "keyword:other": 0.5},
		SignalErrors:      map[string]string{"systemone:risk": "timeout"},
	}
	for name, rule := range map[string]config.RuleNode{
		"not": {Operator: "OR", Conditions: []config.RuleNode{
			{Operator: "NOT", Conditions: []config.RuleNode{failed}},
			present,
		}},
		"and": {Operator: "OR", Conditions: []config.RuleNode{
			{Operator: "AND", Conditions: []config.RuleNode{other, failedMatch}},
			present,
		}},
	} {
		t.Run(name, func(t *testing.T) {
			engine := NewDecisionEngine([]config.Decision{{Name: "route", Rules: rule}}, config.RoutingStrategyPriority)
			result, err := engine.EvaluateDecisionsWithSignals(signals)
			if err != nil || result == nil {
				t.Fatalf("result = %#v, error = %v", result, err)
			}
			if result.Confidence != 0.6 || len(result.MatchedRules) != 1 || result.MatchedRules[0] != "keyword:present" {
				t.Fatalf("confidence = %v, rules = %v", result.Confidence, result.MatchedRules)
			}
		})
	}
}

func TestOperatingPointLabelMatchesKeepUnknownAndLegacyErrorPolicy(t *testing.T) {
	node := config.RuleNode{Type: config.SignalTypeSystemOne, Name: "risk", Label: "unsafe"}
	for _, policy := range []config.UnknownPolicy{config.RuleOnUnknownNoMatch, config.RuleOnUnknownMatch, config.RuleOnUnknownFailRequest} {
		node.OnUnknown = policy
		engine := NewDecisionEngine([]config.Decision{{Name: "route", Rules: node}}, config.RoutingStrategyPriority)
		result, diagnostics, err := engine.EvaluateDecisionsWithDiagnostics(&SignalMatches{SignalErrors: map[string]string{"systemone:risk": "failed"}})
		if (err != nil) != (policy == config.RuleOnUnknownFailRequest) || (result != nil) != (policy == config.RuleOnUnknownMatch) || diagnostics.AppliedUnknownPolicies["route"] != string(policy) {
			t.Fatalf("policy %s result=%+v diagnostics=%+v error=%v", policy, result, diagnostics, err)
		}
	}
	node.OnUnknown = ""
	node.OnError = "match"
	engine := NewDecisionEngine([]config.Decision{{Name: "route", Rules: node}}, config.RoutingStrategyPriority)
	if result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{SignalErrors: map[string]string{"systemone:risk": "failed"}}); err != nil || result == nil {
		t.Fatalf("legacy on_error lost: %+v %v", result, err)
	}
}

// A guard rule that matches by policy, because its model could not read the
// whole input or gave no answer and the rule fails closed, carries no value.
// A condition's own predicate then resolves it as the condition without one.
func TestAPredicateConditionKeepsAGuardsPolicyMatch(t *testing.T) {
	plain := config.RuleNode{Type: config.SignalTypeJailbreak, Name: "guard"}
	strict := plain
	strict.Predicate = &config.NumericPredicate{GTE: float64Ptr(0.95)}
	failClosed := &SignalMatches{
		JailbreakRules:     []string{"guard"},
		SignalErrors:       map[string]string{"jailbreak:guard": "systemone_unavailable"},
		SignalErrorMatches: map[string]bool{"jailbreak:guard": true},
	}
	for name, signals := range map[string]*SignalMatches{
		"unscanned":   {JailbreakRules: []string{"guard"}},
		"fail closed": failClosed,
	} {
		for _, rule := range []config.RuleNode{plain, strict} {
			engine := NewDecisionEngine([]config.Decision{{Name: "block", Rules: rule}}, config.RoutingStrategyPriority)
			if result, err := engine.EvaluateDecisionsWithSignals(signals); err != nil || result == nil {
				t.Errorf("%s, predicate %v: result = %#v, error = %v", name, rule.Predicate != nil, result, err)
			}
		}
	}
	engine := NewDecisionEngine([]config.Decision{{Name: "block", Rules: strict}}, config.RoutingStrategyPriority)
	if result, _ := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		JailbreakRules: []string{"guard"},
		SignalValues:   map[string]float64{"jailbreak:guard": 0.9},
	}); result != nil {
		t.Fatalf("a reported value below the predicate does not match: %#v", result)
	}
	strict.OnUnknown = config.RuleOnUnknownFailRequest
	engine = NewDecisionEngine([]config.Decision{{Name: "block", Rules: strict}}, config.RoutingStrategyPriority)
	if _, err := engine.EvaluateDecisionsWithSignals(failClosed); !errors.Is(err, ErrDecisionUnresolved) {
		t.Fatalf("a fail-closed match stays unknown under rules.on_unknown: %v", err)
	}
}

// NOT reverses a known result only: over a signal that failed, a decision
// without rules.on_unknown does not match.
func TestNOTOverAFailedSignalDoesNotMatchWithoutOnUnknown(t *testing.T) {
	rule := config.RuleNode{Operator: "NOT", Conditions: []config.RuleNode{{Type: config.SignalTypeJailbreak, Name: "prompt_injection"}}}
	engine := NewDecisionEngine([]config.Decision{{Name: "safe_only_route", Rules: rule}}, config.RoutingStrategyPriority)
	result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		SignalErrors: map[string]string{"jailbreak:prompt_injection": "systemone_unavailable"},
	})
	if err != nil || result != nil {
		t.Fatalf("result = %#v, error = %v", result, err)
	}
	if result, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{}); err != nil || result == nil {
		t.Fatalf("a known negative still matches: result = %#v, error = %v", result, err)
	}
}

// A condition's on_error resolves a failed signal, so NOT reverses what it
// resolved to, alone or inside AND and OR; without on_error it stays unknown.
func TestNOTReversesWhatAConditionsOnErrorResolved(t *testing.T) {
	signals := &SignalMatches{SignalErrors: map[string]string{"systemone:spam": "systemone_unavailable"}}
	for _, onError := range []string{"", "no_match", "match"} {
		for _, predicate := range []*config.NumericPredicate{nil, {GTE: float64Ptr(0.5)}} {
			leaf := config.RuleNode{Type: config.SignalTypeSystemOne, Name: "spam", Predicate: predicate, OnError: onError}
			for _, child := range []config.RuleNode{
				leaf,
				{Operator: "AND", Conditions: []config.RuleNode{leaf}},
				{Operator: "OR", Conditions: []config.RuleNode{leaf}},
			} {
				rule := config.RuleNode{Operator: "NOT", Conditions: []config.RuleNode{child}}
				engine := NewDecisionEngine([]config.Decision{{Name: "not_spam", Rules: rule}}, config.RoutingStrategyPriority)
				result, err := engine.EvaluateDecisionsWithSignals(signals)
				if want := onError == "no_match"; err != nil || (result != nil) != want {
					t.Errorf("on_error %q, predicate %t, child %q: result = %#v, error = %v, want a match %t", onError, predicate != nil, child.Operator, result, err, want)
				}
			}
		}
	}
	unresolved := config.RuleNode{Type: config.SignalTypeSystemOne, Name: "spam"}
	resolved := unresolved
	resolved.OnError = "no_match"
	rule := config.RuleNode{Operator: "NOT", Conditions: []config.RuleNode{{Operator: "AND", Conditions: []config.RuleNode{unresolved, resolved}}}}
	engine := NewDecisionEngine([]config.Decision{{Name: "not_spam", Rules: rule}}, config.RoutingStrategyPriority)
	if result, err := engine.EvaluateDecisionsWithSignals(signals); err != nil || result == nil {
		t.Fatalf("a resolved False makes the AND a known negative: result = %#v, error = %v", result, err)
	}
}
