package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// ComplexityVerdictTotal counts verdicts by rule, verdict and the path
	// that produced them. Its main diagnostic value is the shape of the
	// distribution: a remote scorer whose scale does not match a rule's
	// boundaries - a [1,10] model behind hard_above: 0.85, say - shows up as
	// one verdict taking every request, which no per-request log line makes
	// visible.
	ComplexityVerdictTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_complexity_verdict_total",
			Help: "Complexity verdicts by rule, verdict and source path (local, remote_score, remote_labels)",
		},
		[]string{"rule", "verdict", "source"},
	)

	// ComplexityEvaluationFailuresTotal counts evaluations that produced no
	// verdict at all, by the path that failed. A remote scorer outage lands
	// here rather than only in an error log.
	ComplexityEvaluationFailuresTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_complexity_evaluation_failures_total",
			Help: "Complexity evaluations that produced no verdict, by source path",
		},
		[]string{"source"},
	)
)

// RecordComplexityVerdict records one rule's verdict and the path it came from.
func RecordComplexityVerdict(rule, verdict, source string) {
	ComplexityVerdictTotal.WithLabelValues(
		labelOrUnknown(rule), labelOrUnknown(verdict), labelOrUnknown(source)).Inc()
}

// RecordComplexityEvaluationFailure records an evaluation that produced no
// verdict for any rule.
func RecordComplexityEvaluationFailure(source string) {
	ComplexityEvaluationFailuresTotal.WithLabelValues(labelOrUnknown(source)).Inc()
}
