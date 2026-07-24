package main

import (
	"context"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backend"
	backendtrtllm "github.com/vllm-project/semantic-router/src/semantic-router/pkg/backend/tensorrtllm"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// engineTargetBuilder derives adapter targets for one engine kind from the
// router config. New engine adapters (vLLM, SGLang, ATOM) register here.
type engineTargetBuilder struct {
	kind     backend.EngineKind
	register func() error
	targets  func(cfg *config.RouterConfig, ttl, requestTimeout time.Duration) []backend.AdapterTarget
}

// engineTargetBuilders lists the backend telemetry adapters the runtime knows
// how to wire. Each backend_ref self-describes its engine kind and metrics
// surface (option B), so adding an engine is a single entry here.
var engineTargetBuilders = []engineTargetBuilder{
	{
		kind:     backend.EngineKindTensorRTLLM,
		register: backendtrtllm.Register,
		targets: func(cfg *config.RouterConfig, _, _ time.Duration) []backend.AdapterTarget {
			return backendtrtllm.TargetsFromRouterConfig(cfg, backendtrtllm.TargetOptions{})
		},
	},
}

// initializeBackendTelemetryIfEnabled starts the engine-neutral backend
// telemetry collector when enabled. It walks all configured engine adapters,
// builds targets from backend_refs, and runs every adapter behind one Runner.
// The whole path is fail-open: no adapter failure is fatal, and when disabled
// or without targets, router behavior is unchanged.
func initializeBackendTelemetryIfEnabled(cfg *config.RouterConfig, shutdownHooks *[]func()) {
	telemetryCfg := cfg.Observability.BackendTelemetry
	if !telemetryCfg.Enabled {
		return
	}

	pollInterval := durationOrDefault(telemetryCfg.PollInterval, backend.DefaultCollectionInterval)
	ttl := durationOrDefault(telemetryCfg.TTL, backend.DefaultTelemetryTTL)
	requestTimeout := durationOrDefault(telemetryCfg.RequestTimeout, 2*time.Second)

	adapters := make([]backend.TelemetryAdapter, 0, len(engineTargetBuilders))
	engineKinds := make([]string, 0, len(engineTargetBuilders))
	for _, builder := range engineTargetBuilders {
		targets := builder.targets(cfg, ttl, requestTimeout)
		if len(targets) == 0 {
			continue
		}
		if err := builder.register(); err != nil {
			logging.ComponentWarnEvent("router", "backend_telemetry_adapter_register_failed", map[string]interface{}{
				"engine_kind": string(builder.kind),
				"error":       err.Error(),
			})
			continue
		}
		adapter, err := backend.NewAdapter(builder.kind, backend.AdapterConfig{
			Targets:        targets,
			Store:          backend.DefaultStore(),
			Interval:       pollInterval,
			TTL:            ttl,
			RequestTimeout: requestTimeout,
		})
		if err != nil {
			logging.ComponentWarnEvent("router", "backend_telemetry_adapter_create_failed", map[string]interface{}{
				"engine_kind": string(builder.kind),
				"error":       err.Error(),
			})
			continue
		}
		adapters = append(adapters, adapter)
		engineKinds = append(engineKinds, string(builder.kind))
	}

	if len(adapters) == 0 {
		logging.ComponentWarnEvent("router", "backend_telemetry_no_targets", map[string]interface{}{})
		return
	}

	runner, err := backend.NewRunner(backend.RunnerConfig{
		Adapters: adapters,
		Store:    backend.DefaultStore(),
		Interval: pollInterval,
		OnError: func(kind backend.EngineKind, err error) {
			logging.ComponentWarnEvent("router", "backend_telemetry_collect_failed", map[string]interface{}{
				"engine_kind": string(kind),
				"error":       err.Error(),
			})
		},
	})
	if err != nil {
		logging.ComponentWarnEvent("router", "backend_telemetry_runner_create_failed", map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	*shutdownHooks = append(*shutdownHooks, func() {
		cancel()
		logging.ComponentEvent("router", "backend_telemetry_shutdown", map[string]interface{}{})
	})
	go runner.Run(ctx)
	logging.ComponentEvent("router", "backend_telemetry_started", map[string]interface{}{
		"engine_kinds":  engineKinds,
		"adapters":      len(adapters),
		"poll_interval": pollInterval.String(),
		"ttl":           ttl.String(),
	})
}

func durationOrDefault(value string, fallback time.Duration) time.Duration {
	if value == "" {
		return fallback
	}
	parsed, err := time.ParseDuration(value)
	if err != nil || parsed <= 0 {
		return fallback
	}
	return parsed
}
