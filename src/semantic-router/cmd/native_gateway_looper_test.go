package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// A standalone Router runs a Looper decision's model calls in process: the
// backends see every hop, and nothing calls back through the Looper endpoint.
func TestStandaloneRouterRunsLooperHopsInProcess(t *testing.T) {
	var hops atomic.Int32
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model string `json:"model"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		hops.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"c","object":"chat.completion","model":"`+req.Model+`","choices":[{"index":0,`+
			`"message":{"role":"assistant","content":"from `+req.Model+`"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer backend.Close()
	var loopedBack atomic.Int32
	loopback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		loopedBack.Add(1)
		http.Error(w, "the standalone Router must not loop back", http.StatusInternalServerError)
	}))
	defer loopback.Close()

	t.Setenv(configsnapshot.HistoryDirEnv, t.TempDir())
	previous := modelservice.DefaultManager()
	modelservice.SetDefault(nil)
	t.Cleanup(func() { modelservice.SetDefault(previous) })
	document := strings.NewReplacer("BACKEND", strings.TrimPrefix(backend.URL, "http://"), "LOOPER", loopback.URL).Replace(`version: v0.3
listeners:
  - name: http-8899
    address: 127.0.0.1
    port: 8899
providers:
  defaults:
    model: a
  models:
    - name: a
      backend_refs: [{endpoint: BACKEND, protocol: http, provider: vllm}]
    - name: b
      backend_refs: [{endpoint: BACKEND, protocol: http, provider: vllm}]
global:
  stores:
    semantic_cache:
      enabled: false
  integrations:
    looper:
      endpoint: LOOPER/v1/chat/completions
routing:
  modelCards:
    - name: a
    - name: b
  decisions:
    - name: compare
      priority: 10
      rules: {operator: AND, conditions: []}
      modelRefs: [{model: a}, {model: b}]
      algorithm:
        type: ratings
        ratings: {max_concurrent: 2, on_error: fail}
`)
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatal(err)
	}
	server, err := extproc.NewServer(path, 0, false, "", routerruntime.NewRegistry(cfg), extproc.WithConfigParts(upstreamPart(config.GatewayStandalone)))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	served := make(chan error, 1)
	go func() { served <- server.StartContextWithoutExtProc(ctx, nil) }()
	defer func() {
		cancel()
		<-served
	}()
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
	defer front.Close()

	resp, err := http.Post(front.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"vllm-sr/auto","messages":[{"role":"user","content":"compare"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK || !strings.Contains(string(body), "from a") || !strings.Contains(string(body), "from b") {
		t.Fatalf("response %d %s", resp.StatusCode, body)
	}
	if resp.Header.Get("x-vsr-response-path") != "looper" {
		t.Fatalf("response path %q", resp.Header.Get("x-vsr-response-path"))
	}
	if hops.Load() != 2 || loopedBack.Load() != 0 {
		t.Fatalf("backend hops %d, loopback calls %d", hops.Load(), loopedBack.Load())
	}
}
