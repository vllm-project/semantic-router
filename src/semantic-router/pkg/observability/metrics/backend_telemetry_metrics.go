package metrics

import (
	"sync/atomic"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendtelemetry"
)

var backendTelemetryMetrics = &backendTelemetryExporter{
	scrapes: prometheus.NewDesc(
		"llm_backend_telemetry_scrapes_total",
		"Serving-engine metric scrapes by cluster, endpoint, engine and outcome (success, error)",
		[]string{"cluster", "endpoint", "engine", "outcome"}, nil,
	),
	queueDepth: prometheus.NewDesc(
		"llm_backend_telemetry_queue_depth",
		"Requests waiting in a backend's serving engine at its last fresh successful scrape",
		[]string{"cluster", "endpoint"}, nil,
	),
}

func init() {
	prometheus.MustRegister(backendTelemetryMetrics)
}

// BackendTelemetry owns the scrape counters and targets of one telemetry
// generation. Collection can warm it without changing the exported metrics.
type BackendTelemetry struct {
	store     *backendtelemetry.Store
	targets   []backendtelemetry.Target
	endpoints map[backendtelemetry.Identity]string
	scrapes   *prometheus.CounterVec
}

// NewBackendTelemetry creates an unpublished metrics scope for the targets.
func NewBackendTelemetry(store *backendtelemetry.Store, targets []backendtelemetry.Target) *BackendTelemetry {
	endpoints := make(map[backendtelemetry.Identity]string, len(targets))
	for _, target := range targets {
		name := target.EndpointName
		if name == "" {
			name = target.Identity.BackendRef
		}
		endpoints[target.Identity] = name
	}
	return &BackendTelemetry{
		store: store, targets: append([]backendtelemetry.Target(nil), targets...), endpoints: endpoints,
		scrapes: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "llm_backend_telemetry_scrapes_total",
			Help: "Serving-engine metric scrapes by cluster, endpoint, engine and outcome (success, error)",
		}, []string{"cluster", "endpoint", "engine", "outcome"}),
	}
}

// Activate exports only this generation. A rejected candidate never calls it.
func (b *BackendTelemetry) Activate() { backendTelemetryMetrics.active.Store(b) }

// Close withdraws this scope only if it is still active. Draining an older
// generation cannot delete the metrics of its replacement.
func (b *BackendTelemetry) Close() { backendTelemetryMetrics.active.CompareAndSwap(b, nil) }

// RecordScrape records one target's result in this generation's counters.
func (b *BackendTelemetry) RecordScrape(id backendtelemetry.Identity, success bool) {
	endpoint, configured := b.endpoints[id]
	if !configured {
		return
	}
	outcome := "error"
	if success {
		outcome = "success"
	}
	b.scrapes.WithLabelValues(id.Model, endpoint, string(id.Engine), outcome).Inc()
}

type backendTelemetryExporter struct {
	active              atomic.Pointer[BackendTelemetry]
	scrapes, queueDepth *prometheus.Desc
}

func (e *backendTelemetryExporter) Describe(ch chan<- *prometheus.Desc) {
	ch <- e.scrapes
	ch <- e.queueDepth
}

func (e *backendTelemetryExporter) Collect(ch chan<- prometheus.Metric) {
	active := e.active.Load()
	if active == nil {
		return
	}
	active.scrapes.Collect(ch)
	for _, target := range active.targets {
		id := target.Identity
		observation, fresh := active.store.Fresh(id)
		if !fresh {
			continue
		}
		if depth, present := observation.QueueDepth.Get(); present {
			ch <- prometheus.MustNewConstMetric(e.queueDepth, prometheus.GaugeValue, float64(depth), id.Model, active.endpoints[id])
		}
	}
}
