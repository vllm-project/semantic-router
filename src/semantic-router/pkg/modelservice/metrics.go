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
			Help:    "Latency of model runtime calls that reached the runtime, by surface.",
			Buckets: []float64{0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5},
		},
		[]string{"deployment", "surface"},
	)
	transportDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "vsr_model_runtime_transport_seconds",
			Help:    "Per runtime call, the part of the HTTP exchange that carried it outside the runtime: the Router's time for the exchange minus the runtime's Server-Timing total.",
			Buckets: []float64{0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1},
		},
		[]string{"deployment", "surface"},
	)
	serverDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "vsr_model_runtime_server_seconds",
			Help:    "Per runtime call, the runtime's own time for the exchange that carried it, by phase: parse, tokenize, queue, forward, post, serialize and other (the rest of its total).",
			Buckets: []float64{0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5},
		},
		[]string{"deployment", "surface", "phase"},
	)
	cacheTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "vsr_model_runtime_result_cache_total",
			Help: "Classify and decision results served from the per-model result cache (hit) or the runtime (miss).",
		},
		[]string{"deployment", "result"},
	)
	bundleTasks = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "vsr_model_runtime_bundle_tasks",
			Help:    "Tasks per /v1/bundle call.",
			Buckets: []float64{1, 2, 3, 4, 6, 8, 12, 16, 32, 64},
		},
	)
	bundleWait = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "vsr_model_runtime_bundle_wait_seconds",
			Help:    "Time from a bundle's first parked call to its flush.",
			Buckets: []float64{0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005},
		},
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
			Help: "1 while the deployment's model passes its runtime's readiness check.",
		},
		[]string{"deployment"},
	)
	restartsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "vsr_model_runtime_restarts_total",
			Help: "Restarts of the Router-managed runtime process that serves the deployment.",
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
