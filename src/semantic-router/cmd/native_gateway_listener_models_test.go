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

// A keyed listener restricted to the router's auto model: the key is checked
// first, a provider model named by the client never reaches a backend (not
// even through the skip-processing opt-out), the auto model's in-process hops
// still call the provider models, and /v1/models lists only the auto model.
func TestStandaloneListenerServesOnlyItsModels(t *testing.T) {
	var calls atomic.Int32
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model string `json:"model"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		calls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"c","object":"chat.completion","model":"`+req.Model+`","choices":[{"index":0,`+
			`"message":{"role":"assistant","content":"from `+req.Model+`"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer backend.Close()

	t.Setenv(configsnapshot.HistoryDirEnv, t.TempDir())
	previous := modelservice.DefaultManager()
	modelservice.SetDefault(nil)
	t.Cleanup(func() { modelservice.SetDefault(previous) })
	document := strings.ReplaceAll(`version: v0.3
listeners:
  - name: public
    address: 127.0.0.1
    port: 8899
    api_keys: [workshop-key]
    models: [vllm-sr/auto]
providers:
  defaults:
    model: a
  models:
    - name: a
      backend_refs: [{endpoint: BACKEND, protocol: http, provider: vllm}]
    - name: b
      backend_refs: [{endpoint: BACKEND, protocol: http, provider: vllm}]
global:
  router:
    list_backend_models: true
    skip_processing:
      enabled: true
  stores:
    semantic_cache:
      enabled: false
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
`, "BACKEND", strings.TrimPrefix(backend.URL, "http://"))
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}
	cfg, err := config.ParseYAMLBytes([]byte(document))
	if err != nil {
		t.Fatal(err)
	}
	if err = config.ValidateGatewayCapabilities(cfg, config.GatewayStandalone); err != nil {
		t.Fatalf("standalone mode refused the listener's models: %v", err)
	}
	if err = config.ValidateGatewayCapabilities(cfg, config.GatewayExtProc); err == nil ||
		!strings.Contains(err.Error(), "listener 'public': models is unsupported with --gateway extproc") {
		t.Fatalf("extproc mode error = %v, want listener models unsupported", err)
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
	handler, err := gateway.NewHandler(gateway.Options{Serving: nativeServing{pin: server.Pin, engine: engine}, Listener: "public"})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(handler)
	defer front.Close()

	send := func(method, path, key, body string, header map[string]string) (int, string) {
		t.Helper()
		req, err := http.NewRequest(method, front.URL+path, strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		if body != "" {
			req.Header.Set("Content-Type", "application/json")
		}
		if key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
		}
		for name, value := range header {
			req.Header.Set(name, value)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		out, _ := io.ReadAll(resp.Body)
		return resp.StatusCode, string(out)
	}
	chat := func(model string) string {
		return `{"model":"` + model + `","messages":[{"role":"user","content":"compare"}]}`
	}

	if status, _ := send(http.MethodPost, "/v1/chat/completions", "", chat("vllm-sr/auto"), nil); status != http.StatusUnauthorized {
		t.Fatalf("no key: status %d, want 401", status)
	}
	if status, body := send(http.MethodPost, "/v1/chat/completions", "workshop-key", chat("vllm-sr/auto"), nil); status != http.StatusOK {
		t.Fatalf("key + vllm-sr/auto: status %d, want 200: %s", status, body)
	}
	if got := calls.Load(); got != 2 {
		t.Fatalf("the auto model's decision made %d backend calls, want its 2 hops to a and b", got)
	}
	calls.Store(0)
	for _, attempt := range []struct {
		name   string
		body   string
		header map[string]string
		status int
		code   string
	}{
		{name: "provider model", body: chat("a"), status: http.StatusForbidden, code: "model_not_allowed"},
		{
			name: "skip-processing opt-out", body: chat("b"), header: map[string]string{"x-vsr-skip-processing": "true"},
			status: http.StatusForbidden, code: "model_not_allowed",
		},
		{
			name: "model omitted", body: `{"messages":[{"role":"user","content":"hi"}]}`,
			status: http.StatusBadRequest, code: "model_required",
		},
	} {
		status, body := send(http.MethodPost, "/v1/chat/completions", "workshop-key", attempt.body, attempt.header)
		var reply struct {
			Error struct {
				Code string `json:"code"`
			} `json:"error"`
		}
		_ = json.Unmarshal([]byte(body), &reply)
		if status != attempt.status || reply.Error.Code != attempt.code {
			t.Fatalf("%s: status %d body %s, want %d %s", attempt.name, status, body, attempt.status, attempt.code)
		}
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("rejected requests reached a backend %d times", got)
	}

	status, body := send(http.MethodGet, "/v1/models", "workshop-key", "", nil)
	var list struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal([]byte(body), &list); err != nil || status != http.StatusOK {
		t.Fatalf("/v1/models: status %d body %s", status, body)
	}
	if len(list.Data) != 1 || list.Data[0].ID != "vllm-sr/auto" {
		t.Fatalf("/v1/models lists %+v, want only vllm-sr/auto", list.Data)
	}
}
