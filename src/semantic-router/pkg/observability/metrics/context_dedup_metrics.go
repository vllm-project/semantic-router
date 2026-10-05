package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Context-dedup metrics stay low cardinality: the decision name plus bounded
// outcome and reason codes. Conversation content is never a label, and counts
// are emitted independently of Router Replay so a deployment without replay
// capture still observes the action.
var (
	contextDedupEvaluations = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_context_dedup_evaluations_total",
			Help: "Context deduplication evaluations by decision, outcome, and terminal reason.",
		},
		[]string{"decision", "outcome", "reason"},
	)
	contextDedupRemovedTurns = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_context_dedup_removed_turns",
			Help:    "Repeated prior turns removed by a committed context deduplication.",
			Buckets: []float64{1, 2, 4, 8, 16, 32, 64, 128},
		},
		[]string{"decision"},
	)
	contextDedupRemovedMessages = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_context_dedup_removed_messages",
			Help:    "Messages removed by a committed context deduplication.",
			Buckets: []float64{1, 2, 4, 8, 16, 32, 64, 128, 256},
		},
		[]string{"decision"},
	)
	contextDedupRemovedTextBytes = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "llm_context_dedup_removed_text_bytes",
			Help:    "History text bytes removed by a committed context deduplication.",
			Buckets: prometheus.ExponentialBuckets(64, 2, 15),
		},
		[]string{"decision"},
	)
)

// RecordContextDedupEvaluation records one evaluated deduplication policy.
// Removal histograms observe committed removals only, so a proposal that the
// shared executor rejected cannot inflate them.
func RecordContextDedupEvaluation(
	decision string,
	outcome string,
	reason string,
	removedTurns int,
	removedMessages int,
	removedTextBytes int,
) {
	contextDedupEvaluations.WithLabelValues(decision, outcome, reason).Inc()
	if removedMessages <= 0 {
		return
	}
	contextDedupRemovedMessages.WithLabelValues(decision).Observe(float64(removedMessages))
	if removedTurns > 0 {
		contextDedupRemovedTurns.WithLabelValues(decision).Observe(float64(removedTurns))
	}
	if removedTextBytes > 0 {
		contextDedupRemovedTextBytes.WithLabelValues(decision).Observe(float64(removedTextBytes))
	}
}
