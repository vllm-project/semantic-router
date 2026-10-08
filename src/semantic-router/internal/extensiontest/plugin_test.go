// Package extensiontest proves that an extension registered outside the
// Router's code works end to end. Nothing in the Router names the types
// registered here.
package extensiontest

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configschema"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

const stampType = "test_stamp"

// stampPlugin stamps a header on the provider-bound request and on the
// client response. Its fields are its configuration schema.
type stampPlugin struct {
	Header string `json:"header,omitempty"`
	Value  string `json:"value"`
}

func (p *stampPlugin) OnRequest(_ context.Context, request *pluginruntime.PluginRequest) error {
	return request.SetHeader(p.Header, p.Value+" for "+request.Model)
}

func (p *stampPlugin) OnResponse(_ context.Context, response *pluginruntime.PluginResponse) error {
	return response.SetHeader(p.Header, fmt.Sprintf("%s %d", p.Value, response.Status))
}

func init() {
	if err := config.RegisterDecisionPlugin(config.NewDecisionPluginType(
		config.DecisionPluginCatalogEntry{
			Type: stampType, DisplayName: "Test Stamp", Description: "Stamp a header on the request and the response.",
		},
		config.PluginOptions[stampPlugin]{
			Strict: true,
			Defaults: func(plugin *stampPlugin) {
				if plugin.Header == "" {
					plugin.Header = "x-test-stamp"
				}
			},
			Validate: func(at config.PluginAt, plugin *stampPlugin) error {
				if plugin.Value == "" {
					return at.Errorf("value is required")
				}
				return nil
			},
		},
	)); err != nil {
		panic(err)
	}
}

func stampDocument(backend, configuration string) string {
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
      backend_refs:
        - endpoint: ` + strings.TrimPrefix(backend, "http://") + `
          protocol: http
          provider: vllm
global:
  stores:
    semantic_cache:
      enabled: false
routing:
  modelCards:
    - name: m
  signals:
    keywords:
      - name: greeting
        operator: OR
        keywords: ["hello"]
  decisions:
    - name: greet
      priority: 10
      rules:
        operator: AND
        conditions:
          - type: keyword
            name: greeting
      modelRefs:
        - model: m
          use_reasoning: false
      plugins:
        - type: ` + stampType + `
          configuration:
` + configuration
}

func writeDocument(t *testing.T, document string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestARegisteredPluginRunsOnTheRequestAndTheResponse(t *testing.T) {
	var mu sync.Mutex
	var stamped string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		mu.Lock()
		stamped = r.Header.Get("x-test-stamp")
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"c","object":"chat.completion","model":"m","choices":[{"index":0,`+
			`"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],`+
			`"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer backend.Close()
	router, err := extproc.NewOpenAIRouter(writeDocument(t, stampDocument(backend.URL, "            value: stamped\n")))
	if err != nil {
		t.Fatal(err)
	}
	defer router.Close()
	set, err := upstream.Build(router.Config, upstream.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = set.Close(ctx)
	}()
	engine := routing.DefaultOptions
	engine.ExecutesFallback = true
	handler, err := gateway.NewHandler(gateway.Options{
		Serving:  gateway.Static(gateway.Serving{Engine: routing.NewEngine(extproc.NewRouterService(router), engine), Upstream: set}),
		Listener: "http-8899",
	})
	if err != nil {
		t.Fatal(err)
	}
	front := httptest.NewServer(handler)
	defer front.Close()

	resp, err := http.Post(front.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"vllm-sr/auto","messages":[{"role":"user","content":"hello there"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
	mu.Lock()
	defer mu.Unlock()
	if resp.StatusCode != http.StatusOK || stamped != "stamped for m" || resp.Header.Get("x-test-stamp") != "stamped 200" {
		t.Fatalf("status %d, upstream saw %q, client saw %q", resp.StatusCode, stamped, resp.Header.Get("x-test-stamp"))
	}
}

func TestARegisteredPluginOwnsItsSchemaDefaultsAndValidation(t *testing.T) {
	if !config.IsSupportedDecisionPluginType(stampType) {
		t.Fatal("the registered type is not supported")
	}
	for name, configuration := range map[string]string{
		"validator":     "            value: \"\"\n",
		"strict schema": "            value: stamped\n            colour: red\n",
	} {
		if _, err := config.ParseYAMLBytes([]byte(stampDocument("http://127.0.0.1:1", configuration))); err == nil {
			t.Fatalf("%s: an invalid payload loaded", name)
		} else if !strings.Contains(err.Error(), stampType) {
			t.Fatalf("%s: the error does not name the plugin: %v", name, err)
		}
	}
	cfg, err := config.ParseYAMLBytes([]byte(stampDocument("http://127.0.0.1:1", "            value: stamped\n")))
	if err != nil {
		t.Fatal(err)
	}
	payload, err := config.DecodeDecisionPluginAt(config.PluginAt{Decision: "greet", Type: stampType}, cfg.Decisions[0].Plugins[0])
	if err != nil || payload.(*stampPlugin).Header != "x-test-stamp" {
		t.Fatalf("decoded %+v (%v), want the default header", payload, err)
	}
	schema, err := configschema.GenerateFromSource("../../../..")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(schema), `"`+stampType+`"`) {
		t.Fatal("the generated schema does not list the registered plugin type")
	}
}
