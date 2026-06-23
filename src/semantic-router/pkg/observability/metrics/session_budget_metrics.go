package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
)

// SessionBudgetEvaluations counts every session token-budget evaluation, labeled
// by the graduated stage it selected (none, shape_tools, compress, downgrade,
// terminate) so operators can see how often enforcement fires and at which
// severity.
var SessionBudgetEvaluations = promauto.NewCounterVec(
	prometheus.CounterOpts{
		Name: "vsr_session_budget_evaluations_total",
		Help: "Total session token-budget evaluations, labeled by selected graduated stage.",
	},
	[]string{"stage"},
)

// SessionBudgetRatio tracks the distribution of the over-budget ratio
// (cumulative tokens / configured budget) observed at evaluation time, so
// operators can tune budget_tokens and the stage thresholds against real
// traffic.
var SessionBudgetRatio = promauto.NewHistogram(
	prometheus.HistogramOpts{
		Name:    "vsr_session_budget_ratio",
		Help:    "Distribution of session cumulative-tokens / budget ratio at evaluation time.",
		Buckets: []float64{0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0},
	},
)

// RecordSessionBudgetStage increments the evaluation counter for one budget
// evaluation. Empty stage falls back to consts.UnknownLabel.
func RecordSessionBudgetStage(stage string) {
	if stage == "" {
		stage = consts.UnknownLabel
	}
	SessionBudgetEvaluations.WithLabelValues(stage).Inc()
}

// ObserveSessionBudgetRatio records one over-budget ratio observation.
func ObserveSessionBudgetRatio(ratio float64) {
	SessionBudgetRatio.Observe(ratio)
}
