//go:build !windows

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

// A listener whose Chat API routes unserved models answers a model name a
// client picks itself, such as one of its provider's own models, and a
// provider model the listener does not list, as if they named its auto
// model; its Messages API, without the setting, refuses them as before.
func TestStandaloneListenerRoutesUnservedModels(t *testing.T) {
	var calls atomic.Int32
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"c","object":"chat.completion","model":"a","choices":[{"index":0,`+
			`"message":{"role":"assistant","content":"from a"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer backend.Close()

	t.Setenv(configsnapshot.HistoryDirEnv, t.TempDir())
	previous := modelservice.DefaultManager()
	modelservice.SetDefault(nil)
	t.Cleanup(func() { modelservice.SetDefault(previous) })
	document := strings.ReplaceAll(`version: v0.5
listeners:
  - name: public
    address: 127.0.0.1
    port: 8899
    apis:
      chat_completions:
        models: [vllm-sr/auto]
        unserved_models: {policy: route, route_to: vllm-sr/auto}
      messages: {models: [vllm-sr/auto]}
providers:
  defaults:
    model: a
  models:
    - name: a
      backend_refs: [{endpoint: BACKEND, protocol: http, provider: vllm}]
recipes:
  - name: default
    decisions:
      - name: answer
        priority: 10
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: a}]
`, "BACKEND", strings.TrimPrefix(backend.URL, "http://"))
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatal(err)
	}
	if err = config.ValidateGatewayCapabilities(cfg, config.GatewayExtProc); err == nil ||
		!strings.Contains(err.Error(), "apis.chat_completions.unserved_models is unsupported with --gateway extproc") {
		t.Fatalf("extproc mode error = %v, want unserved_models unsupported", err)
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
	handler, err := gateway.NewHandler(gateway.Options{Serving: nativeServing{pin: server.Pin, engine: routing.DefaultOptions}, Listener: "public"})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(handler)
	defer front.Close()

	send := func(path, body string) (int, string, string) {
		t.Helper()
		resp, err := http.Post(front.URL+path, "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		out, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, string(out), resp.Header.Get("x-vsr-selected-model")
	}
	chat := func(model string) string {
		return `{"model":"` + model + `","messages":[{"role":"user","content":"hello"}]}`
	}
	for _, model := range []string{"gpt-5.6-sol", "a", "vllm-sr/auto"} {
		status, body, selected := send("/v1/chat/completions", chat(model))
		if status != http.StatusOK || selected != "a" {
			t.Fatalf("chat with %s: status %d, selected %q, body %s; want 200 from a", model, status, selected, body)
		}
	}
	if got := calls.Load(); got != 3 {
		t.Fatalf("the routed requests made %d backend calls, want 3", got)
	}

	status, body, _ := send("/v1/messages", `{"model":"gpt-5.6-sol","max_tokens":8,"messages":[{"role":"user","content":"hello"}]}`)
	var refused struct {
		Error struct {
			Type string `json:"type"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(body), &refused); err != nil || status != http.StatusNotFound {
		t.Fatalf("messages without the setting: status %d body %s, want 404", status, body)
	}
	if got := calls.Load(); got != 3 {
		t.Fatalf("the refused request reached a backend")
	}
}
