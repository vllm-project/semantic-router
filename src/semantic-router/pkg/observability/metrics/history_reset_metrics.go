package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// History-reset metrics stay low cardinality: the decision name plus bounded
// outcome and reason codes. Recovery keys, signal payloads, and conversation
// content are never labels, and counts are emitted independently of Router
// Replay so a deployment without replay capture still observes the action.
var (
	historyResetEvaluations = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_history_reset_evaluations_total",
			Help: "History-reset evaluations by decision, outcome, and terminal reason.",
		},
		[]string{"decision", "outcome", "reason"},
	)
	historyResetRemovedTurns = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_history_reset_removed_turns",
			Help:    "Complete prior turns removed by a committed history reset.",
			Buckets: []float64{1, 2, 4, 8, 16, 32, 64, 128},
		},
		[]string{"decision"},
	)
	historyResetRemovedMessages = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_history_reset_removed_messages",
			Help:    "Messages removed by a committed history reset.",
			Buckets: []float64{1, 2, 4, 8, 16, 32, 64, 128, 256},
		},
		[]string{"decision"},
	)
	historyResetRecovery = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_history_reset_recovery_total",
			Help: "Recoverable history-reset removals by status.",
		},
		[]string{"status"},
	)
)

// RecordHistoryResetEvaluation records one evaluated reset policy. Removal
// histograms observe committed removals only, so a proposal that the shared
// executor rejected cannot inflate them.
func RecordHistoryResetEvaluation(
	decision string,
	outcome string,
	reason string,
	removedTurns int,
	removedMessages int,
) {
	historyResetEvaluations.WithLabelValues(decision, outcome, reason).Inc()
	if removedMessages > 0 {
		historyResetRemovedMessages.WithLabelValues(decision).Observe(float64(removedMessages))
	}
	if removedTurns > 0 {
		historyResetRemovedTurns.WithLabelValues(decision).Observe(float64(removedTurns))
	}
}

// RecordHistoryResetRecovery records what happened to the recoverable copy of
// the removed history.
func RecordHistoryResetRecovery(status string) {
	if status == "" {
		return
	}
	historyResetRecovery.WithLabelValues(status).Inc()
}
