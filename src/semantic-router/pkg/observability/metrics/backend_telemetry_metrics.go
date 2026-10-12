package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Backend telemetry labels are bounded by configuration: cluster is a provider
// model alias and endpoint one of its backend ref names.
var (
	// BackendTelemetryScrapesTotal counts collector reads of serving-engine
	// metrics by outcome (success, error).
	BackendTelemetryScrapesTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "llm_backend_telemetry_scrapes_total",
			Help: "Serving-engine metric scrapes by cluster, endpoint, engine and outcome (success, error)",
		},
		[]string{"cluster", "endpoint", "engine", "outcome"},
	)

	// BackendTelemetryQueueDepth is the last queue depth an engine reported.
	BackendTelemetryQueueDepth = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "llm_backend_telemetry_queue_depth",
			Help: "Requests waiting in a backend's serving engine at its last successful scrape",
		},
		[]string{"cluster", "endpoint"},
	)
)

// RecordBackendTelemetryScrape records one scrape of one backend.
func RecordBackendTelemetryScrape(cluster, endpoint, engine, outcome string) {
	BackendTelemetryScrapesTotal.WithLabelValues(
		labelOrUnknown(cluster), labelOrUnknown(endpoint), labelOrUnknown(engine), labelOrUnknown(outcome),
	).Inc()
}

// SetBackendTelemetryQueueDepth records the queue depth an engine reported.
func SetBackendTelemetryQueueDepth(cluster, endpoint string, depth float64) {
	BackendTelemetryQueueDepth.WithLabelValues(labelOrUnknown(cluster), labelOrUnknown(endpoint)).Set(depth)
}
