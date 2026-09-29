package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var requestDemandTokens = promauto.NewHistogramVec(
	prometheus.HistogramOpts{
		Name:    "llm_request_demand_tokens",
		Help:    "Observed request token demand by lifecycle stage, accounting source, and component.",
		Buckets: prometheus.ExponentialBuckets(128, 2, 14),
	},
	[]string{"stage", "counting_source", "component"},
)

// RecordRequestDemand records bounded, content-free request-demand evidence.
// Stage, source, and component are closed vocabularies owned by the caller.
func RecordRequestDemand(stage, countingSource string, prompt, reservedOutput, total int, totalKnown bool) {
	stage = labelOrUnknown(stage)
	countingSource = labelOrUnknown(countingSource)
	requestDemandTokens.WithLabelValues(stage, countingSource, "prompt").Observe(float64(prompt))
	requestDemandTokens.WithLabelValues(stage, countingSource, "reserved_output").Observe(float64(reservedOutput))
	if totalKnown {
		requestDemandTokens.WithLabelValues(stage, countingSource, "total").Observe(float64(total))
	}
}
