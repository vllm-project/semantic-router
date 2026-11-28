package metrics

import (
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendtelemetry"
)

func telemetryScope(t *testing.T, depth int64) (*BackendTelemetry, backendtelemetry.Observation) {
	t.Helper()
	o := backendtelemetry.Observation{
		Schema:   backendtelemetry.SchemaVersion,
		Identity: backendtelemetry.Identity{Model: "m", BackendRef: "gpu-a", Engine: backendtelemetry.EngineVLLM},
		Source:   "test", CollectedAt: time.Now(), TTL: time.Minute, QueueDepth: backendtelemetry.Observed(depth),
	}
	store := backendtelemetry.NewStore(0)
	if err := store.Upsert(o); err != nil {
		t.Fatal(err)
	}
	return NewBackendTelemetry(store, []backendtelemetry.Target{{Identity: o.Identity, EndpointName: "m_gpu-a"}}), o
}

func gatherTelemetry(t *testing.T, registry *prometheus.Registry) map[string]*dto.MetricFamily {
	t.Helper()
	families, err := registry.Gather()
	if err != nil {
		t.Fatal(err)
	}
	out := make(map[string]*dto.MetricFamily, len(families))
	for _, family := range families {
		out[family.GetName()] = family
	}
	return out
}

func assertTelemetryDepth(t *testing.T, registry *prometheus.Registry, want float64) {
	t.Helper()
	family := gatherTelemetry(t, registry)["llm_backend_telemetry_queue_depth"]
	if family == nil || len(family.Metric) != 1 || family.Metric[0].GetGauge().GetValue() != want {
		t.Fatalf("queue-depth metrics = %v, want only %v", family, want)
	}
}

func TestBackendTelemetryMetricsFollowActivationAndOwnership(t *testing.T) {
	registry := prometheus.NewRegistry()
	registry.MustRegister(backendTelemetryMetrics)
	old, observation := telemetryScope(t, 9)
	old.RecordScrape(observation.Identity, true)
	defer old.Close()
	if len(gatherTelemetry(t, registry)) != 0 {
		t.Fatal("a warming candidate exported metrics")
	}
	old.Activate()
	assertTelemetryDepth(t, registry, 9)

	discarded, _ := telemetryScope(t, 100)
	discarded.RecordScrape(observation.Identity, true)
	discarded.Close()
	assertTelemetryDepth(t, registry, 9)

	next, _ := telemetryScope(t, 0)
	defer next.Close()
	next.RecordScrape(observation.Identity, true)
	next.Activate()
	old.RecordScrape(observation.Identity, false)
	old.Close()
	assertTelemetryDepth(t, registry, 0)
	family := gatherTelemetry(t, registry)["llm_backend_telemetry_scrapes_total"]
	if family == nil || len(family.Metric) != 1 || family.Metric[0].GetCounter().GetValue() != 1 {
		t.Fatalf("the old collector changed new counters: %v", family)
	}
	next.Close()
	if len(gatherTelemetry(t, registry)) != 0 {
		t.Fatal("closed telemetry left exported series")
	}
}

func TestBackendTelemetryMetricsOmitMissingAndExpiredQueues(t *testing.T) {
	registry := prometheus.NewRegistry()
	registry.MustRegister(backendTelemetryMetrics)
	scope, o := telemetryScope(t, 4)
	defer scope.Close()
	scope.RecordScrape(o.Identity, true)
	scope.Activate()
	assertTelemetryDepth(t, registry, 4)
	o.QueueDepth = backendtelemetry.Count{}
	if err := scope.store.Upsert(o); err != nil {
		t.Fatal(err)
	}
	if gatherTelemetry(t, registry)["llm_backend_telemetry_queue_depth"] != nil {
		t.Fatal("missing queue retained its old gauge")
	}
	o.QueueDepth = backendtelemetry.Observed(int64(4))
	o.TTL = time.Nanosecond
	if err := scope.store.Upsert(o); err != nil {
		t.Fatal(err)
	}
	if gatherTelemetry(t, registry)["llm_backend_telemetry_queue_depth"] != nil {
		t.Fatal("expired queue stayed exported as current load")
	}
	if gatherTelemetry(t, registry)["llm_backend_telemetry_scrapes_total"] == nil {
		t.Fatal("expiry removed collection-health counters")
	}

	disabled := NewBackendTelemetry(nil, nil)
	defer disabled.Close()
	disabled.Activate()
	scope.Close()
	if len(gatherTelemetry(t, registry)) != 0 {
		t.Fatal("disabling telemetry retained old series")
	}
}

func TestBackendTelemetryEndpointLabelsMatchUpstream(t *testing.T) {
	scope, o := telemetryScope(t, 4)
	defer scope.Close()
	scope.Activate()
	registry := prometheus.NewRegistry()
	registry.MustRegister(backendTelemetryMetrics, UpstreamActiveRequests)
	AddUpstreamActiveRequests(o.Identity.Model, "m_gpu-a", 1)
	defer DeleteUpstreamCluster(o.Identity.Model)
	families := gatherTelemetry(t, registry)
	labelValue := func(family *dto.MetricFamily) string {
		for _, label := range family.Metric[0].Label {
			if label.GetName() == "endpoint" {
				return label.GetValue()
			}
		}
		return ""
	}
	telemetry := labelValue(families["llm_backend_telemetry_queue_depth"])
	upstream := labelValue(families["llm_upstream_active_requests"])
	if telemetry != upstream || telemetry != "m_gpu-a" {
		t.Fatalf("endpoint labels differ: telemetry %q, upstream %q", telemetry, upstream)
	}
}

func TestDeleteUpstreamClusterIncludesLoadAwarePicks(t *testing.T) {
	name := "removed-load-aware"
	RecordUpstreamLoadAwarePick(name, "telemetry")
	DeleteUpstreamCluster(name)
	registry := prometheus.NewRegistry()
	registry.MustRegister(UpstreamLoadAwarePicksTotal)
	families, err := registry.Gather()
	if err != nil {
		t.Fatal(err)
	}
	for _, family := range families {
		for _, metric := range family.Metric {
			for _, label := range metric.Label {
				if label.GetName() == "cluster" && label.GetValue() == name {
					t.Fatal("removed cluster retained load-aware pick metrics")
				}
			}
		}
	}
}
