package selection

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// ModelSelectionFallbackTotal tracks fallback outcomes separately from
// successful selector choices. Labels are bounded runtime reason codes.
var ModelSelectionFallbackTotal *prometheus.CounterVec

// RecordSelection records a basic model selection event with tier label.
func RecordSelection(
	method string,
	decision string,
	model string,
	tier AlgorithmTier,
	score float64,
) {
	if !metricsEnabled {
		return
	}
	tierStr := string(tier)
	ModelSelectionTotal.WithLabelValues(method, model, decision, tierStr).Inc()
	ModelSelectionScore.WithLabelValues(method, model).Observe(score)
	ModelSelectionHistory.WithLabelValues(method, decision).Inc()
}

// RecordSelectionDuration observes selector latency for successful and failed
// attempts.
func RecordSelectionDuration(
	method SelectionMethod,
	tier AlgorithmTier,
	duration time.Duration,
) {
	if !metricsEnabled {
		return
	}
	ModelSelectionDuration.WithLabelValues(
		string(method),
		string(tier),
	).Observe(duration.Seconds())
}

// RecordSelectionFallback increments the bounded fallback outcome counter.
func RecordSelectionFallback(
	method SelectionMethod,
	decision string,
	reason string,
) {
	if !metricsEnabled {
		return
	}
	ModelSelectionFallbackTotal.WithLabelValues(
		string(method),
		decision,
		reason,
	).Inc()
}
