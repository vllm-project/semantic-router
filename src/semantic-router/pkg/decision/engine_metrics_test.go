package decision

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	dto "github.com/prometheus/client_model/go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

func histogramSampleCount(t *testing.T, h prometheus.Histogram) uint64 {
	t.Helper()
	var m dto.Metric
	if err := h.Write(&m); err != nil {
		t.Fatalf("histogram Write: %v", err)
	}
	if m.Histogram == nil {
		t.Fatalf("histogram payload missing")
	}
	return m.Histogram.GetSampleCount()
}

// TestEvaluateDecisionsWithSignals_EmitsMetricsExactlyOnce locks in the
// invariant that the canonical recording site for the decision-evaluation
// histogram and decision-match counter is the engine itself. If a caller
// (notably the extproc runtime) ever re-introduces wrapper-level recording,
// these counts would double and this test will fail.
//
// Regression guard for the double-counting bug previously present in
// pkg/extproc/req_filter_classification_runtime.go::runDecisionEngine.
func TestEvaluateDecisionsWithSignals_EmitsMetricsExactlyOnce(t *testing.T) {
	engine := NewDecisionEngine(
		nil,
		nil,
		nil,
		[]config.Decision{
			{
				Name:     "catch-all",
				Priority: 10,
				Rules: config.RuleCombination{
					Operator:   "AND",
					Conditions: []config.RuleCondition{},
				},
			},
		},
		"priority",
	)

	latencyBefore := histogramSampleCount(t, metrics.DecisionEvaluationLatency)
	matchesBefore := testutil.ToFloat64(metrics.DecisionMatchTotal.WithLabelValues("catch-all"))

	if _, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{}); err != nil {
		t.Fatalf("EvaluateDecisionsWithSignals returned unexpected error: %v", err)
	}

	latencyAfter := histogramSampleCount(t, metrics.DecisionEvaluationLatency)
	matchesAfter := testutil.ToFloat64(metrics.DecisionMatchTotal.WithLabelValues("catch-all"))

	if got := latencyAfter - latencyBefore; got != 1 {
		t.Fatalf("DecisionEvaluationLatency observation delta = %d, want 1 (double-counting bug?)", got)
	}
	if got := matchesAfter - matchesBefore; got != 1 {
		t.Fatalf("DecisionMatchTotal{decision_name=\"catch-all\"} delta = %v, want 1 (double-counting bug?)", got)
	}
}

func TestEvaluateDecisionsRecordsUnknownPolicy(t *testing.T) {
	engine := NewDecisionEngine(nil, nil, nil, []config.Decision{{Name: "guarded", Rules: config.RuleNode{
		Type:      config.SignalTypeClassifier,
		Name:      "risk",
		Label:     "RISKY",
		Predicate: &config.NumericPredicate{GTE: float64Ptr(0.5)},
		OnUnknown: config.RuleOnUnknownNoMatch,
	}}}, config.RoutingStrategyPriority)
	counter := metrics.DecisionUnknownTotal.WithLabelValues("guarded", string(config.RuleOnUnknownNoMatch))
	before := testutil.ToFloat64(counter)

	if _, err := engine.EvaluateDecisionsWithSignals(&SignalMatches{
		SignalErrors: map[string]string{"classifier:risk": "timeout"},
	}); err != nil {
		t.Fatal(err)
	}

	if got := testutil.ToFloat64(counter) - before; got != 1 {
		t.Fatalf("DecisionUnknownTotal delta = %v, want 1", got)
	}
}

func TestDecisionConfidenceMetricsRespectScoredStatus(t *testing.T) {
	for _, test := range []struct {
		name    string
		rule    config.RuleNode
		signals *SignalMatches
		scored  bool
	}{
		{
			name:    "keyword",
			rule:    config.RuleNode{Type: "keyword", Name: "marker"},
			signals: &SignalMatches{KeywordRules: []string{"marker"}},
		},
		{
			name: "on_unknown_match",
			rule: config.RuleNode{
				Type: config.SignalTypeClassifier, Name: "risk", Label: "RISKY",
				Predicate: &config.NumericPredicate{GTE: float64Ptr(0.5)}, OnUnknown: config.RuleOnUnknownMatch,
			},
			signals: &SignalMatches{SignalErrors: map[string]string{"classifier:risk": "unavailable"}},
		},
		{
			name:    "catch_all",
			rule:    config.RuleNode{Operator: "AND"},
			signals: &SignalMatches{},
			scored:  true,
		},
		{
			name:    "predicate",
			rule:    config.RuleNode{Type: "structure", Name: "size", Predicate: &config.NumericPredicate{GTE: float64Ptr(1)}},
			signals: &SignalMatches{SignalValues: map[string]float64{"structure:size": 2}},
		},
		{
			name: "reported_zero",
			rule: config.RuleNode{Type: "domain", Name: "example"},
			signals: &SignalMatches{
				DomainRules:       []string{"example"},
				SignalConfidences: map[string]float64{"domain:example": 0},
			},
			scored: true,
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			name := t.Name()
			engine := NewDecisionEngine(nil, nil, nil, []config.Decision{{Name: name, Rules: test.rule}}, config.RoutingStrategyPriority)
			histogram := metrics.DecisionConfidence.WithLabelValues(name).(prometheus.Histogram)
			before := histogramSampleCount(t, histogram)
			matchesBefore := testutil.ToFloat64(metrics.DecisionMatchTotal.WithLabelValues(name))
			result, err := engine.EvaluateDecisionsWithSignals(test.signals)
			if err != nil || result == nil || result.ConfidenceScored != test.scored {
				t.Fatalf("decision = %#v, error = %v; scored = %v", result, err, test.scored)
			}
			want := uint64(0)
			if test.scored && !result.CatchAll {
				want = 1
			}
			if got := histogramSampleCount(t, histogram) - before; got != want {
				t.Fatalf("confidence observations = %d, want %d", got, want)
			}
			if got := testutil.ToFloat64(metrics.DecisionMatchTotal.WithLabelValues(name)) - matchesBefore; got != 1 {
				t.Fatalf("match counter = %g, want 1", got)
			}
		})
	}
}
