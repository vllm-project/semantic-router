package metrics

import (
	"strconv"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// topicContinuityEvaluations uses enum labels only, so its lifetime
	// cardinality is fixed regardless of operator-configured rule names.
	topicContinuityEvaluations = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_topic_continuity_evaluations_total",
			Help: "Topic-continuity rule evaluations by class, reason, coverage, and fallback.",
		},
		[]string{"class", "reason", "coverage", "fallback"},
	)
	topicContinuityPreparedInputBytes = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "llm_topic_continuity_prepared_input_bytes",
			Help:    "Bytes of history evidence prepared per topic-continuity history policy.",
			Buckets: prometheus.ExponentialBuckets(1024, 2, 11),
		},
	)
)

// RecordTopicContinuityEvaluation counts one rule's outcome. Labels are bounded
// enums only; the rule name is deliberately not a label.
func RecordTopicContinuityEvaluation(class, reason, coverage string, fallback bool) {
	topicContinuityEvaluations.WithLabelValues(class, reason, coverage, strconv.FormatBool(fallback)).Inc()
}

// RecordTopicContinuityPreparation observes the evidence bytes of one
// preparation; rules sharing a history policy share one observation.
func RecordTopicContinuityPreparation(inputBytes int) {
	topicContinuityPreparedInputBytes.Observe(float64(inputBytes))
}
