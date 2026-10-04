package modelservice

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	requestsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "vsr_model_runtime_requests_total",
			Help: "Calls to model runtime deployments by outcome (ok, timeout, unavailable, overloaded, rejected, failed).",
		},
		[]string{"deployment", "outcome"},
	)
	requestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "vsr_model_runtime_request_duration_seconds",
			Help:    "Latency of model runtime calls that reached the runtime.",
			Buckets: []float64{0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5},
		},
		[]string{"deployment"},
	)
	unknownAnswers = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "vsr_model_runtime_unknown_answers_total",
			Help: "Questions left unknown (fail-open) by reason.",
		},
		[]string{"deployment", "reason"},
	)
	readyGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "vsr_model_runtime_ready",
			Help: "1 while the deployment's runtime passes its readiness check.",
		},
		[]string{"deployment"},
	)
	restartsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "vsr_model_runtime_restarts_total",
			Help: "Restarts of Router-managed runtime processes.",
		},
		[]string{"deployment"},
	)
)

// RecordUnknown counts questions that ended unknown for a reason.
func RecordUnknown(deployment, reason string, questions int) {
	if questions > 0 {
		unknownAnswers.WithLabelValues(deployment, reason).Add(float64(questions))
	}
}
