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

// A ratings decision answers with one choice per candidate model. Chat
// Completions carries them; Messages and Responses cannot, so a request on
// either is refused with a 400 before any candidate is called, rather than
// failing with a 502 after every candidate answered.
func TestStandaloneRatingsAnswersOnlyAnAPIThatCarriesItsChoices(t *testing.T) {
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
	document := strings.ReplaceAll(`version: v0.5
listeners:
  - name: public
    address: 127.0.0.1
    port: 8899
    apis: {chat_completions: {}, messages: {}, responses: {}}
providers:
  defaults:
    model: a
  models:
    - name: a
      backend_refs: [{endpoint: BACKEND, protocol: http, provider: vllm}]
    - name: b
      backend_refs: [{endpoint: BACKEND, protocol: http, provider: vllm}]
recipes:
  - name: default
    decisions:
      - name: compare
        priority: 10
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: a}, {model: b}]
        algorithm:
          type: ratings
          on_error: fail
          ratings: {max_concurrent: 2}
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
`, "BACKEND", strings.TrimPrefix(backend.URL, "http://"))
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
	handler, err := gateway.NewHandler(gateway.Options{Serving: nativeServing{pin: server.Pin, engine: routing.DefaultOptions}, Listener: "public"})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(handler)
	defer front.Close()

	for _, api := range []struct {
		path, body string
		status     int
		calls      int32
		answer     string
	}{
		{"/v1/chat/completions", `{"model":"vllm-sr/auto","messages":[{"role":"user","content":"compare"}]}`, http.StatusOK, 2, `"index":1`},
		{
			"/v1/messages", `{"model":"vllm-sr/auto","max_tokens":16,"messages":[{"role":"user","content":"compare"}]}`, http.StatusBadRequest, 0,
			"which the Anthropic Messages API cannot return; call the Chat Completions API",
		},
		{
			"/v1/responses", `{"model":"vllm-sr/auto","store":false,"input":"compare"}`, http.StatusBadRequest, 0,
			"which the Responses API cannot return; call the Chat Completions API",
		},
	} {
		calls.Store(0)
		resp, err := http.Post(front.URL+api.path, "application/json", strings.NewReader(api.body))
		if err != nil {
			t.Fatal(err)
		}
		body, _ := io.ReadAll(resp.Body)
		_ = resp.Body.Close()
		if resp.StatusCode != api.status || calls.Load() != api.calls || !strings.Contains(string(body), api.answer) {
			t.Fatalf("%s: status %d after %d backend calls, body %s; want %d after %d calls with %q",
				api.path, resp.StatusCode, calls.Load(), body, api.status, api.calls, api.answer)
		}
	}
}
