package decision

import (
	"errors"
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
	traceResult, traces, _, _ := engine.EvaluateDecisionsWithTraceAndDiagnostics(signals)
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

func TestOnUnknownPolicies(t *testing.T) {
	rule := config.RuleNode{
		Type:      config.SignalTypeClassifier,
		Name:      "risk",
		Label:     "RISKY",
		Predicate: &config.NumericPredicate{GTE: float64Ptr(0.5)},
		OnError:   "match",
	}
	signals := &SignalMatches{SignalErrors: map[string]string{"classifier:risk": "timeout"}}
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
			engine := NewDecisionEngine(nil, nil, nil, []config.Decision{{Name: "route", Rules: rule}}, config.RoutingStrategyPriority)
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
		Type:      config.SignalTypeClassifier,
		Name:      "risk",
		Label:     "RISKY",
		Predicate: &config.NumericPredicate{GTE: float64Ptr(0.5)},
		OnError:   "match",
	}
	engine := NewDecisionEngine(nil, nil, nil, []config.Decision{{Name: "route", Rules: rule}}, config.RoutingStrategyPriority)
	result, traces, diagnostics, err := engine.EvaluateDecisionsWithTraceAndDiagnostics(&SignalMatches{
		SignalErrors: map[string]string{"classifier:risk": "timeout"},
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
		Type:      config.SignalTypeClassifier,
		Name:      "risk",
		Label:     "RISKY",
		Predicate: &config.NumericPredicate{GTE: float64Ptr(0.5)},
		OnUnknown: config.RuleOnUnknownFailRequest,
	}
	matching := config.RuleNode{Type: config.SignalTypeKeyword, Name: "present"}
	engine := NewDecisionEngine(nil, nil, nil, []config.Decision{
		{Name: "guarded", Rules: unresolved},
		{Name: "route", Rules: matching},
	}, config.RoutingStrategyPriority)
	result, traces, _, err := engine.EvaluateDecisionsWithTraceAndDiagnostics(&SignalMatches{
		KeywordRules: []string{"present"},
		SignalErrors: map[string]string{"classifier:risk": "timeout"},
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
		Type:      config.SignalTypeClassifier,
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
		SignalErrors: map[string]string{"classifier:risk": "timeout"},
	}
	for name, decisions := range map[string][]config.Decision{
		"guarded first": {guarded, good},
		"good first":    {good, guarded},
	} {
		t.Run(name, func(t *testing.T) {
			engine := NewDecisionEngine(nil, nil, nil, decisions, config.RoutingStrategyPriority)
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

func TestUnknownTraceRecordsErrorAndPolicy(t *testing.T) {
	rule := config.RuleNode{
		Type:      config.SignalTypeClassifier,
		Name:      "risk",
		Label:     "RISKY",
		Predicate: &config.NumericPredicate{GTE: float64Ptr(0.5)},
		OnUnknown: config.RuleOnUnknownNoMatch,
	}
	engine := NewDecisionEngine(nil, nil, nil, []config.Decision{{Name: "route", Rules: rule}}, config.RoutingStrategyPriority)
	_, traces, diagnostics, err := engine.EvaluateDecisionsWithTraceAndDiagnostics(&SignalMatches{
		SignalErrors: map[string]string{"classifier:risk": "timeout"},
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
		Type:      config.SignalTypeClassifier,
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
		SignalErrors:      map[string]string{"classifier:risk": "timeout"},
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
			engine := NewDecisionEngine(nil, nil, nil, []config.Decision{{Name: "route", Rules: rule}}, config.RoutingStrategyPriority)
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
