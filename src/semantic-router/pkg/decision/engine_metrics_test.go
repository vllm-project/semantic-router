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
