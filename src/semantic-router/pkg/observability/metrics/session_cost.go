package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
)

// SessionTurnCost tracks per-turn cost for sessions with pricing configured, labeled by
// model, VSR domain/category, and currency so non-USD deployments are represented correctly.
var SessionTurnCost = promauto.NewHistogramVec(
	prometheus.HistogramOpts{
		Name:    "llm_session_turn_cost",
		Help:    "Distribution of per-turn cost attributed to a logical session (model + domain category + currency). Only recorded when pricing is configured.",
		Buckets: []float64{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0},
	},
	[]string{"model", "domain", "currency"},
)

// RecordSessionTurnCost records the per-turn cost histogram for sessions with pricing
// configured. Callers must only invoke this when pricing is active (cost == 0 for a
// free model is valid; cost == 0 because pricing is absent should not be recorded).
func RecordSessionTurnCost(model, domain, currency string, cost float64) {
	if model == "" {
		model = consts.UnknownLabel
	}
	if domain == "" {
		domain = consts.UnknownLabel
	}
	if currency == "" {
		currency = "USD"
	}
	SessionTurnCost.WithLabelValues(model, domain, currency).Observe(cost)
}
