package main

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendtelemetry"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendtelemetry/vllm"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// backendTelemetryStore holds the latest load observation of every backend
// for the life of the process; collectors of successive snapshots write to it.
var backendTelemetryStore = backendtelemetry.NewStore(0)

// backendTelemetryRegistry lists the engines the Router can read load
// metrics from.
func backendTelemetryRegistry() *backendtelemetry.Registry {
	registry := backendtelemetry.NewRegistry()
	if err := vllm.Register(registry); err != nil {
		panic(err) // a duplicate registration is a programming error
	}
	return registry
}

// backendTelemetryPart makes each snapshot own a collector that reads the
// load metrics of its backends into store, the one store the process keeps
// across snapshots. A snapshot that leaves the backends and settings unchanged
// keeps the running collector.
func backendTelemetryPart(store *backendtelemetry.Store) configsnapshot.PartBuilder {
	registry := backendTelemetryRegistry()
	return configsnapshot.PartBuilder{
		Component: configsnapshot.ComponentBackendTelemetry,
		Build: func(_ context.Context, candidate *configsnapshot.Snapshot, _ configsnapshot.Part) (configsnapshot.Part, error) {
			return startBackendTelemetry(candidate.Config(), registry, store)
		},
	}
}

// startBackendTelemetry starts a collector for the backends of cfg that name
// an engine. Disabled collection, or no such backend, starts an idle one.
func startBackendTelemetry(
	cfg *config.RouterConfig,
	registry *backendtelemetry.Registry,
	store *backendtelemetry.Store,
) (*backendtelemetry.Collector, error) {
	telemetry := cfg.Observability.BackendTelemetry
	if !telemetry.Enabled {
		return backendtelemetry.StartCollector(backendtelemetry.CollectorOptions{}), nil
	}
	settings, err := telemetry.Settings()
	if err != nil {
		return nil, err
	}
	byEngine := backendTelemetryTargets(cfg.VLLMEndpoints)
	var adapters []backendtelemetry.Adapter
	targetCount := 0
	for _, engine := range backendtelemetry.Engines() {
		targets := byEngine[engine]
		if len(targets) == 0 {
			continue
		}
		if !registry.Has(engine) {
			logging.ComponentWarnEvent("router", "backend_telemetry_engine_unsupported", map[string]interface{}{
				"engine":   string(engine),
				"backends": len(targets),
				"effect":   "these backends report no load metrics; load-aware balancing falls back for their models",
			})
			continue
		}
		adapter, err := registry.New(engine, backendtelemetry.AdapterConfig{
			Targets: targets, RequestTimeout: settings.RequestTimeout, TTL: settings.TTL,
		})
		if err != nil {
			return nil, err
		}
		adapters = append(adapters, adapter)
		targetCount += len(targets)
	}
	logging.ComponentEvent("router", "backend_telemetry_started", map[string]interface{}{
		"backends":      targetCount,
		"poll_interval": settings.PollInterval.String(),
		"ttl":           settings.TTL.String(),
	})
	return backendtelemetry.StartCollector(backendtelemetry.CollectorOptions{
		Adapters: adapters,
		Store:    store,
		Interval: settings.PollInterval,
		Report:   reportBackendTelemetry(byEngine),
	}), nil
}

// backendTelemetryTargets groups the endpoints that name an engine by it.
func backendTelemetryTargets(endpoints []config.VLLMEndpoint) map[backendtelemetry.Engine][]backendtelemetry.Target {
	byEngine := make(map[backendtelemetry.Engine][]backendtelemetry.Target)
	for _, endpoint := range endpoints {
		if !endpoint.Telemetry.Enabled() {
			continue
		}
		identity := endpoint.TelemetryIdentity()
		byEngine[identity.Engine] = append(byEngine[identity.Engine], backendtelemetry.Target{
			Identity: identity, MetricsURL: endpoint.MetricsURL(),
		})
	}
	return byEngine
}

// reportBackendTelemetry records each round: a target with an observation
// succeeded, every other target failed.
func reportBackendTelemetry(
	byEngine map[backendtelemetry.Engine][]backendtelemetry.Target,
) func(backendtelemetry.Engine, []backendtelemetry.Observation, error) {
	return func(engine backendtelemetry.Engine, observations []backendtelemetry.Observation, err error) {
		observed := make(map[backendtelemetry.Identity]bool, len(observations))
		for _, o := range observations {
			observed[o.Identity] = true
			if depth, ok := o.QueueDepth.Get(); ok {
				metrics.SetBackendTelemetryQueueDepth(o.Identity.Model, o.Identity.BackendRef, float64(depth))
			}
		}
		for _, target := range byEngine[engine] {
			outcome := "success"
			if !observed[target.Identity] {
				outcome = "error"
			}
			metrics.RecordBackendTelemetryScrape(target.Identity.Model, target.Identity.BackendRef, string(engine), outcome)
		}
		if err != nil {
			logging.ComponentDebugEvent("router", "backend_telemetry_scrape_failed", map[string]interface{}{
				"engine": string(engine), "error": err.Error(),
			})
		}
	}
}
