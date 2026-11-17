package main

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
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

// A retired generation closes the connections its scrapes kept alive, once
// its last round has ended, instead of leaving them to the idle timeout.
func TestBackendTelemetryCloseReleasesItsConnections(t *testing.T) {
	var open atomic.Int64
	engine := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("# TYPE vllm:num_requests_waiting gauge\nvllm:num_requests_waiting 1\n"))
	}))
	engine.Config.ConnState = func(_ net.Conn, state http.ConnState) {
		switch state {
		case http.StateNew:
			open.Add(1)
		case http.StateClosed, http.StateHijacked:
			open.Add(-1)
		}
	}
	engine.Start()
	defer engine.Close()
	cfg := telemetryConfig(t, strings.TrimPrefix(engine.URL, "http://"), "        enabled: true\n        poll_interval: 20ms")
	part, err := startBackendTelemetry(cfg, backendTelemetryRegistry())
	if err != nil {
		t.Fatal(err)
	}
	id := backendtelemetry.Identity{Model: "m", BackendRef: "gpu-a", Engine: backendtelemetry.EngineVLLM}
	deadline := time.Now().Add(5 * time.Second)
	for _, fresh := part.store.Fresh(id); !fresh; _, fresh = part.store.Fresh(id) {
		if time.Now().After(deadline) {
			t.Fatal("no observation was collected")
		}
		time.Sleep(5 * time.Millisecond)
	}
	if err := part.Close(context.Background()); err != nil {
		t.Fatal(err)
	}
	deadline = time.Now().Add(5 * time.Second)
	for open.Load() != 0 {
		if time.Now().After(deadline) {
			t.Fatalf("%d scrape connections stayed open after the generation closed", open.Load())
		}
		time.Sleep(5 * time.Millisecond)
	}
}
