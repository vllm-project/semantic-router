package metrics

import (
	"math"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var requestOutcomes = promauto.NewCounterVec(prometheus.CounterOpts{
	Name: "llm_request_outcomes_total",
	Help: "Client request terminal outcomes, counted once; catalog and inference traffic are distinct.",
}, []string{"traffic_kind", "outcome"})

var requestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
	Name:    "llm_request_duration_seconds",
	Help:    "Client request duration from request headers to terminal response or interruption, including streaming.",
	Buckets: []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800},
}, []string{"traffic_kind", "outcome"})

func RecordRequestOutcome(kind, outcome string, seconds float64) {
	switch kind {
	case "inference", "inference_internal", "catalog", "response_object", "health", "other":
	default:
		kind = "other"
	}
	switch outcome {
	case "success", "client_error", "server_error", "canceled", "timeout", "incomplete", "error":
	default:
		outcome = "error"
	}
	requestOutcomes.WithLabelValues(kind, outcome).Inc()
	if seconds > 0 && !math.IsNaN(seconds) && !math.IsInf(seconds, 0) {
		requestDuration.WithLabelValues(kind, outcome).Observe(seconds)
	}
}
