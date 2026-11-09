package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendtelemetry"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// fakeVLLM serves a chat completion that names the backend and the queue
// depth gauge vLLM exports; metricsDown makes /metrics fail.
type fakeVLLM struct {
	name        string
	waiting     atomic.Int64
	metricsDown atomic.Bool
	served      atomic.Int64
	server      *httptest.Server
}

func newFakeVLLM(t *testing.T, name string, waiting int64) *fakeVLLM {
	t.Helper()
	f := &fakeVLLM{name: name}
	f.waiting.Store(waiting)
	f.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/metrics" {
			if f.metricsDown.Load() {
				http.Error(w, "metrics unavailable", http.StatusServiceUnavailable)
				return
			}
			_, _ = fmt.Fprintf(w, "# TYPE vllm:num_requests_waiting gauge\nvllm:num_requests_waiting{model_name=\"m\"} %d\n"+
				"# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running{model_name=\"m\"} 1\n", f.waiting.Load())
			return
		}
		_, _ = io.Copy(io.Discard, r.Body)
		f.served.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Backend", f.name)
		_, _ = io.WriteString(w, `{"id":"c","object":"chat.completion","model":"m","choices":[{"index":0,`+
			`"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(f.server.Close)
	return f
}

func (f *fakeVLLM) hostPort() string { return strings.TrimPrefix(f.server.URL, "http://") }

func loadAwareDocument(a, b *fakeVLLM) string {
	return `version: v0.3
listeners:
  - name: http-8899
    address: 127.0.0.1
    port: 8899
providers:
  defaults:
    model: m
  models:
    - name: m
      reliability:
        lb_policy: load_aware
      backend_refs:
        - name: gpu-a
          endpoint: ` + a.hostPort() + `
          protocol: http
          provider: vllm
          engine: vllm
        - name: gpu-b
          endpoint: ` + b.hostPort() + `
          protocol: http
          provider: vllm
          engine: vllm
global:
  services:
    observability:
      backend_telemetry:
        enabled: true
        poll_interval: 20ms
        ttl: 200ms
  stores:
    semantic_cache:
      enabled: false
routing:
  modelCards:
    - name: m
`
}

// waitForQueueDepth waits until this generation's store holds a fresh
// observation of a backend with the given queue depth.
func waitForQueueDepth(t *testing.T, store *backendtelemetry.Store, ref string, depth int64) {
	t.Helper()
	id := backendtelemetry.Identity{Model: "m", BackendRef: ref, Engine: backendtelemetry.EngineVLLM}
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if o, ok := store.Fresh(id); ok {
			if got, _ := o.QueueDepth.Get(); got == depth {
				return
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("%s never reported a fresh queue depth of %d", ref, depth)
}

func waitUntilStale(t *testing.T, store *backendtelemetry.Store, ref string) {
	t.Helper()
	id := backendtelemetry.Identity{Model: "m", BackendRef: ref, Engine: backendtelemetry.EngineVLLM}
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if _, ok := store.Fresh(id); !ok {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("%s telemetry never went stale", ref)
}

func servedBy(t *testing.T, front string, n int) map[string]int {
	t.Helper()
	counts := map[string]int{}
	for range n {
		resp, err := http.Post(front+"/v1/chat/completions", "application/json",
			strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hi"}]}`))
		if err != nil {
			t.Fatal(err)
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d", resp.StatusCode)
		}
		counts[resp.Header.Get("X-Backend")]++
	}
	return counts
}

func newLoadAwareGateway(t *testing.T, a, b *fakeVLLM) (*extproc.Server, *httptest.Server) {
	t.Helper()
	t.Setenv(configsnapshot.HistoryDirEnv, t.TempDir())
	previous := modelservice.DefaultManager()
	modelservice.SetDefault(nil)
	t.Cleanup(func() { modelservice.SetDefault(previous) })
	document := loadAwareDocument(a, b)
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatal(err)
	}
	server, err := extproc.NewServer(path, 0, false, "", routerruntime.NewRegistry(cfg),
		extproc.WithConfigParts(backendTelemetryPart(), upstreamPart(config.GatewayStandalone)))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	served := make(chan error, 1)
	go func() { served <- server.StartContextWithoutExtProc(ctx, nil) }()
	t.Cleanup(func() {
		cancel()
		select {
		case serveErr := <-served:
			if serveErr != nil {
				t.Error(serveErr)
			}
		case <-time.After(10 * time.Second):
			t.Error("gateway did not stop")
		}
	})
	if err = server.WaitForServing(ctx); err != nil {
		t.Fatal(err)
	}
	engine := routing.DefaultOptions
	engine.ExecutesFallback = true
	handler, err := gateway.NewHandler(gateway.Options{Serving: nativeServing{pin: server.Pin, engine: engine}, Listener: "http-8899"})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(handler)
	t.Cleanup(front.Close)
	return server, front
}

// Drives the standalone serving path, including live engine queue changes
// and conservative fallback when a backend stops exporting metrics.
func TestNativeGatewayLoadAwareRoutesByEngineQueueDepth(t *testing.T) {
	a, b := newFakeVLLM(t, "gpu-a", 50), newFakeVLLM(t, "gpu-b", 0)
	server, front := newLoadAwareGateway(t, a, b)
	lease, err := server.Pin()
	if err != nil {
		t.Fatal(err)
	}
	store := lease.Snapshot.Part(configsnapshot.ComponentBackendTelemetry).(*backendTelemetryRuntime).store
	lease.Release()
	waitForQueueDepth(t, store, "gpu-a", 50)
	waitForQueueDepth(t, store, "gpu-b", 0)
	if counts := servedBy(t, front.URL, 20); counts["gpu-b"] != 20 {
		t.Fatalf("served = %v, want every request on gpu-b, whose engine queue is empty", counts)
	}

	a.waiting.Store(0)
	b.waiting.Store(50)
	waitForQueueDepth(t, store, "gpu-a", 0)
	waitForQueueDepth(t, store, "gpu-b", 50)
	if counts := servedBy(t, front.URL, 20); counts["gpu-a"] != 20 {
		t.Fatalf("served = %v, want every request on gpu-a once the queues swap", counts)
	}

	b.metricsDown.Store(true)
	waitUntilStale(t, store, "gpu-b")
	if counts := servedBy(t, front.URL, 60); counts["gpu-a"] == 0 || counts["gpu-b"] == 0 {
		t.Fatalf("served = %v, want least_request to use both backends once gpu-b's telemetry is stale", counts)
	}
}
