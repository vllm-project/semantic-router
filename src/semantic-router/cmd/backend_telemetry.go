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

func backendTelemetryRegistry() *backendtelemetry.Registry {
	registry := backendtelemetry.NewRegistry()
	if err := vllm.Register(registry); err != nil {
		panic(err) // a duplicate registration is a programming error
	}
	return registry
}

// backendTelemetryRuntime owns a generation's collector, samples and exported
// metrics. Snapshots with unchanged telemetry inputs share the whole part.
type backendTelemetryRuntime struct {
	store     *backendtelemetry.Store
	collector *backendtelemetry.Collector
	metrics   *metrics.BackendTelemetry
}

func (p *backendTelemetryRuntime) Activate() { p.metrics.Activate() }

func (p *backendTelemetryRuntime) Close(ctx context.Context) error {
	p.metrics.Close()
	err := p.collector.Close(ctx)
	clearSamples := func() {
		if p.store != nil {
			p.store.Clear()
		}
	}
	if err != nil {
		// A bounded drain may end before an adapter returns. Clear only after
		// the writer has stopped, so a late result cannot repopulate the store.
		cleanupContext := context.WithoutCancel(ctx)
		go func() {
			_ = p.collector.Close(cleanupContext)
			clearSamples()
		}()
	} else {
		clearSamples()
	}
	return err
}

func backendTelemetryPart() configsnapshot.PartBuilder {
	registry := backendTelemetryRegistry()
	return configsnapshot.PartBuilder{
		Component: configsnapshot.ComponentBackendTelemetry,
		Build: func(_ context.Context, candidate *configsnapshot.Snapshot, _ configsnapshot.Part) (configsnapshot.Part, error) {
			return startBackendTelemetry(candidate.Config(), registry)
		},
	}
}

func startBackendTelemetry(cfg *config.RouterConfig, registry *backendtelemetry.Registry) (*backendTelemetryRuntime, error) {
	telemetry := cfg.Observability.BackendTelemetry
	if !telemetry.Enabled {
		return &backendTelemetryRuntime{
			collector: backendtelemetry.StartCollector(backendtelemetry.CollectorOptions{}),
			metrics:   metrics.NewBackendTelemetry(nil, nil),
		}, nil
	}
	settings, err := telemetry.Settings()
	if err != nil {
		return nil, err
	}
	byEngine := backendTelemetryTargets(cfg.VLLMEndpoints)
	var adapters []backendtelemetry.Adapter
	var targets []backendtelemetry.Target
	for _, engine := range backendtelemetry.Engines() {
		engineTargets := byEngine[engine]
		if len(engineTargets) == 0 {
			continue
		}
		if !registry.Has(engine) {
			logging.ComponentWarnEvent("router", "backend_telemetry_engine_unsupported", map[string]interface{}{
				"engine":   string(engine),
				"backends": len(engineTargets),
				"effect":   "these backends report no load metrics; load-aware balancing falls back for their models",
			})
			continue
		}
		adapter, err := registry.New(engine, backendtelemetry.AdapterConfig{
			Targets: engineTargets, RequestTimeout: settings.RequestTimeout, TTL: settings.TTL,
		})
		if err != nil {
			return nil, err
		}
		adapters = append(adapters, adapter)
		targets = append(targets, engineTargets...)
	}
	part := &backendTelemetryRuntime{store: backendtelemetry.NewStore(settings.TTL)}
	part.metrics = metrics.NewBackendTelemetry(part.store, targets)
	part.collector = backendtelemetry.StartCollector(backendtelemetry.CollectorOptions{
		Adapters: adapters, Store: part.store, Interval: settings.PollInterval,
		Report: reportBackendTelemetry(byEngine, part.metrics),
	})
	logging.ComponentEvent("router", "backend_telemetry_started", map[string]interface{}{
		"backends": len(targets), "poll_interval": settings.PollInterval.String(), "ttl": settings.TTL.String(),
	})
	return part, nil
}

func backendTelemetryTargets(endpoints []config.VLLMEndpoint) map[backendtelemetry.Engine][]backendtelemetry.Target {
	byEngine := make(map[backendtelemetry.Engine][]backendtelemetry.Target)
	for _, endpoint := range endpoints {
		if !endpoint.Telemetry.Enabled() {
			continue
		}
		identity := endpoint.TelemetryIdentity()
		byEngine[identity.Engine] = append(byEngine[identity.Engine], backendtelemetry.Target{
			Identity: identity, EndpointName: endpoint.Name, MetricsURL: endpoint.MetricsURL(),
		})
	}
	return byEngine
}

func reportBackendTelemetry(
	byEngine map[backendtelemetry.Engine][]backendtelemetry.Target,
	scope *metrics.BackendTelemetry,
) func(backendtelemetry.Engine, []backendtelemetry.Observation, error) {
	return func(engine backendtelemetry.Engine, observations []backendtelemetry.Observation, err error) {
		observed := make(map[backendtelemetry.Identity]bool, len(observations))
		for _, o := range observations {
			observed[o.Identity] = o.Validate() == nil
		}
		for _, target := range byEngine[engine] {
			scope.RecordScrape(target.Identity, observed[target.Identity])
		}
		if err != nil {
			logging.ComponentDebugEvent("router", "backend_telemetry_scrape_failed", map[string]interface{}{
				"engine": string(engine), "error": err.Error(),
			})
		}
	}
}
