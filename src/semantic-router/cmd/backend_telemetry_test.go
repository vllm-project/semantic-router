package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendtelemetry"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func telemetryConfig(t *testing.T, hostPort, telemetry string) *config.RouterConfig {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(`version: v0.3
listeners:
  - name: http-8899
    address: 127.0.0.1
    port: 8899
providers:
  defaults:
    model: m
  models:
    - name: m
      backend_refs:
        - name: gpu-a
          endpoint: ` + hostPort + `
          protocol: http
          provider: vllm
          engine: vllm
        - name: gpu-b
          endpoint: 127.0.0.1:9
          protocol: http
          provider: vllm
global:
  services:
    observability:
      backend_telemetry:
` + telemetry + `
  stores:
    semantic_cache:
      enabled: false
routing:
  modelCards:
    - name: m
`))
	if err != nil {
		t.Fatal(err)
	}
	return cfg
}

func TestBackendTelemetryCollectsFromEngineBackends(t *testing.T) {
	engine := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/metrics" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte("# TYPE vllm:num_requests_waiting gauge\nvllm:num_requests_waiting 4\n"))
	}))
	defer engine.Close()
	cfg := telemetryConfig(t, strings.TrimPrefix(engine.URL, "http://"), "        enabled: true\n        poll_interval: 20ms")

	targets := backendTelemetryTargets(cfg.VLLMEndpoints)
	if len(targets) != 1 || len(targets[backendtelemetry.EngineVLLM]) != 1 {
		t.Fatalf("targets = %+v, want only the backend that names an engine", targets)
	}

	store := backendtelemetry.NewStore(0)
	collector, err := startBackendTelemetry(cfg, backendTelemetryRegistry(), store)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = collector.Close(context.Background()) }()
	id := backendtelemetry.Identity{Model: "m", BackendRef: "gpu-a", Engine: backendtelemetry.EngineVLLM}
	deadline := time.Now().Add(5 * time.Second)
	for {
		if o, ok := store.Fresh(id); ok {
			if depth, _ := o.QueueDepth.Get(); depth != 4 {
				t.Fatalf("queue depth = %d, want 4", depth)
			}
			if o.TTL != 60*time.Millisecond {
				t.Fatalf("ttl = %v, want three poll intervals", o.TTL)
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("no observation was collected")
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func TestBackendTelemetryDisabledCollectsNothing(t *testing.T) {
	scraped := make(chan struct{}, 1)
	engine := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		select {
		case scraped <- struct{}{}:
		default:
		}
	}))
	defer engine.Close()
	cfg := telemetryConfig(t, strings.TrimPrefix(engine.URL, "http://"), "        enabled: false")
	collector, err := startBackendTelemetry(cfg, backendTelemetryRegistry(), backendtelemetry.NewStore(0))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = collector.Close(context.Background()) }()
	select {
	case <-scraped:
		t.Fatal("a disabled collector scraped a backend")
	case <-time.After(50 * time.Millisecond):
	}
}

func TestBackendTelemetryRejectsInvalidSettings(t *testing.T) {
	_, err := config.ParseYAMLBytes([]byte(strings.ReplaceAll(
		`version: v0.3
providers:
  defaults:
    model: m
  models:
    - name: m
      backend_refs:
        - endpoint: 127.0.0.1:8000
          provider: vllm
global:
  services:
    observability:
      backend_telemetry:
        enabled: true
        poll_interval: 5s
        ttl: 1s
routing:
  modelCards:
    - name: m
`, "\t", "  ")))
	if err == nil || !strings.Contains(err.Error(), "ttl must not be shorter than poll_interval") {
		t.Fatalf("ParseYAMLBytes() = %v", err)
	}
}
