package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Shadow dispatch results. Every shadow observation ends in exactly one of
// these so operators can account for sampled, dropped, failed, and completed
// calls without reading logs.
const (
	ShadowDispatchResultCompleted  = "completed"
	ShadowDispatchResultFailed     = "failed"
	ShadowDispatchResultDropped    = "dropped"
	ShadowDispatchResultSampledOut = "sampled_out"
)

var (
	// ShadowDispatchTotal counts shadow observations by decision, result, and
	// the deterministic reason code recorded on the replay outcome. Reason
	// values come from a fixed set so cardinality stays bounded.
	ShadowDispatchTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "sr_shadow_dispatch_total",
			Help: "Total number of shadow model dispatch observations by decision, result, and reason",
		},
		[]string{"decision", "result", "reason"},
	)

	// ShadowDispatchLatency observes end-to-end shadow call latency, excluding
	// queue wait, for completed and failed calls.
	ShadowDispatchLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "sr_shadow_dispatch_latency_seconds",
			Help:    "Latency of shadow model calls in seconds, excluding queue wait",
			Buckets: []float64{0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60},
		},
		[]string{"decision"},
	)

	// ShadowDispatchInflight tracks shadow calls currently executing.
	ShadowDispatchInflight = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "sr_shadow_dispatch_inflight",
			Help: "Number of shadow model calls currently in flight",
		},
		[]string{"decision"},
	)

	// ShadowDispatchQueued tracks shadow calls waiting for an in-flight slot.
	ShadowDispatchQueued = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "sr_shadow_dispatch_queued",
			Help: "Number of shadow model calls waiting for a concurrency slot",
		},
		[]string{"decision"},
	)
)

// RecordShadowDispatch records one terminal shadow observation.
func RecordShadowDispatch(decision, result, reason string) {
	ShadowDispatchTotal.WithLabelValues(decision, result, reason).Inc()
}

// RecordShadowDispatchLatency records the execution latency of one shadow call.
func RecordShadowDispatchLatency(decision string, seconds float64) {
	ShadowDispatchLatency.WithLabelValues(decision).Observe(seconds)
}
