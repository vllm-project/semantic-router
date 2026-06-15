package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/consts"
)

// CacheWarmthEstimate tracks the distribution of KV-cache warmth probability estimates
// per model so operators can correlate cache-hit rates with latency trends.
var CacheWarmthEstimate = promauto.NewHistogramVec(
	prometheus.HistogramOpts{
		Name:    "llm_cache_warmth_estimate",
		Help:    "Distribution of KV-cache warmth probability estimates per model (0=cold, 1=warm).",
		Buckets: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9},
	},
	[]string{"model"},
)

// SessionModelTransitions counts in-session model switches so operators can track
// how often routing decisions change mid-conversation.
var SessionModelTransitions = promauto.NewCounterVec(
	prometheus.CounterOpts{
		Name: "llm_session_model_transitions_total",
		Help: "Total in-session model switches observed, labeled by source and destination model.",
	},
	[]string{"from_model", "to_model"},
)

// RecordCacheWarmthEstimate records one warmth estimate observation for model.
// No-ops for out-of-range values and falls back to the unknown label when model is empty.
func RecordCacheWarmthEstimate(model string, warmth float64) {
	if warmth < 0 || warmth > 1 {
		return
	}
	if model == "" {
		model = consts.UnknownLabel
	}
	CacheWarmthEstimate.WithLabelValues(model).Observe(warmth)
}

// RecordSessionModelTransition increments the transition counter for a from→to model pair.
func RecordSessionModelTransition(fromModel, toModel string) {
	if fromModel == "" {
		fromModel = consts.UnknownLabel
	}
	if toModel == "" {
		toModel = consts.UnknownLabel
	}
	SessionModelTransitions.WithLabelValues(fromModel, toModel).Inc()
}
