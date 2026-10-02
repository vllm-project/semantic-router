package metrics

import (
	"math"
	"slices"

	"github.com/prometheus/client_golang/prometheus"
)

var (
	requestTrafficKinds     = []string{"inference", "inference_internal", "catalog", "response_object", "health", "other"}
	requestTerminalOutcomes = []string{"success", "client_error", "server_error", "canceled", "timeout", "incomplete", "error"}
)

type requestOutcomeMetrics struct {
	outcomes *prometheus.CounterVec
	duration *prometheus.HistogramVec
}

var requestMetrics = newRequestOutcomeMetrics(prometheus.DefaultRegisterer)

func newRequestOutcomeMetrics(registerer prometheus.Registerer) *requestOutcomeMetrics {
	m := &requestOutcomeMetrics{
		outcomes: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "llm_request_outcomes_total",
			Help: "Client request terminal outcomes, counted once; catalog and inference traffic are distinct.",
		}, []string{"traffic_kind", "outcome"}),
		duration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "llm_request_duration_seconds",
			Help:    "Client request duration from request headers to terminal response or interruption, including streaming.",
			Buckets: []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800},
		}, []string{"traffic_kind", "outcome"}),
	}
	// A scrape before the first request must expose a zero baseline for increase/rate.
	// Creating empty histogram children does not add latency observations.
	for _, kind := range requestTrafficKinds {
		for _, outcome := range requestTerminalOutcomes {
			m.outcomes.WithLabelValues(kind, outcome)
			m.duration.WithLabelValues(kind, outcome)
		}
	}
	registerer.MustRegister(m.outcomes, m.duration)
	return m
}

func RecordRequestOutcome(kind, outcome string, seconds float64) {
	requestMetrics.record(kind, outcome, seconds)
}

func (m *requestOutcomeMetrics) record(kind, outcome string, seconds float64) {
	if !slices.Contains(requestTrafficKinds, kind) {
		kind = "other"
	}
	if !slices.Contains(requestTerminalOutcomes, outcome) {
		outcome = "error"
	}
	m.outcomes.WithLabelValues(kind, outcome).Inc()
	if seconds > 0 && !math.IsNaN(seconds) && !math.IsInf(seconds, 0) {
		m.duration.WithLabelValues(kind, outcome).Observe(seconds)
	}
}
