package main

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendtelemetry"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

type lateTelemetryAdapter struct {
	started     chan struct{}
	finish      chan struct{}
	observation backendtelemetry.Observation
}

func (*lateTelemetryAdapter) Engine() backendtelemetry.Engine { return backendtelemetry.EngineVLLM }

func (a *lateTelemetryAdapter) Collect(context.Context) ([]backendtelemetry.Observation, error) {
	close(a.started)
	<-a.finish
	return []backendtelemetry.Observation{a.observation}, nil
}

func TestBackendTelemetryBoundedCloseRejectsLateSamples(t *testing.T) {
	observation := backendtelemetry.Observation{
		Schema:   backendtelemetry.SchemaVersion,
		Identity: backendtelemetry.Identity{Model: "m", BackendRef: "gpu-a", Engine: backendtelemetry.EngineVLLM},
		Source:   "test", CollectedAt: time.Now(), QueueDepth: backendtelemetry.Observed(int64(4)),
	}
	adapter := &lateTelemetryAdapter{started: make(chan struct{}), finish: make(chan struct{}), observation: observation}
	store := backendtelemetry.NewStore(0)
	if err := store.Upsert(observation); err != nil {
		t.Fatal(err)
	}
	part := &backendTelemetryRuntime{
		store:   store,
		metrics: metrics.NewBackendTelemetry(store, []backendtelemetry.Target{{Identity: observation.Identity}}),
		collector: backendtelemetry.StartCollector(backendtelemetry.CollectorOptions{
			Adapters: []backendtelemetry.Adapter{adapter}, Store: store, Interval: time.Hour,
		}),
	}
	<-adapter.started
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if err := part.Close(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("bounded Close() = %v", err)
	}
	close(adapter.finish)
	waitForTelemetryCleanup(t, store)
	if _, held := store.Get(observation.Identity); held {
		t.Fatal("a late scrape repopulated the retired generation")
	}
}
