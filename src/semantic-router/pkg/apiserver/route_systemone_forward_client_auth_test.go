//go:build !windows

package apiserver

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
)

// forwardClientAuthConfig publishes the native entrypoint native/auto on the
// public listener.
const forwardClientAuthConfig = `
version: v0.5
listeners:
  - name: public
    port: 8801
    apis:
      systemone:
        schema: vllm-sr/4.0
        models: [native/auto]
providers:
  models:
    - name: kai
      api_format: systemone
      deployment: local-kai
global:
  model_catalog:
    deployments:
      local-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: cpu
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
  - api: systemone
    model_names: [native/auto]
    recipe: native
recipes:
  - name: default
  - name: native
    decisions:
      - name: classify
        rules: {}
        modelRefs: [{model: kai}]
        algorithm:
          type: escalate
          budget: {deadline: 3s, max_calls: 1}
          escalate:
            quality:
              type: uncalibrated
              accept:
                native:
                  - {question_type: choice, field: top_probability, gte: 0.9}
            stages:
              - {name: fast, model: kai}
`

// A forwarded System One request is admitted like one on a listener: a
// protected entrypoint needs one of its keys as a Bearer token, and a reload
// that protects the entrypoint after the forward retained its generation, but
// before the request's model is known, refuses the request.
func TestSystemOneForwardAdmitsEntrypointClientKeys(t *testing.T) {
	t.Setenv("NATIVE_CLIENT_KEY", "native-1")
	snapshot := func(protect bool) *configsnapshot.Snapshot {
		cfg, err := config.ParseYAMLBytes([]byte(forwardClientAuthConfig))
		if err != nil {
			t.Fatal(err)
		}
		if protect {
			// The parser does not accept client_auth until every path enforces it.
			cfg.Entrypoints[1].ClientAuth = &config.EntrypointClientAuth{Type: config.ClientAuthBearerAPIKey, APIKeyEnvs: []string{"NATIVE_CLIENT_KEY"}}
		}
		installed, err := configsnapshot.NewManager(configsnapshot.Options{}).Install(context.Background(), configsnapshot.Update{Config: cfg})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = installed.Release(context.Background()) })
		return installed
	}
	public, protected := snapshot(false), snapshot(true)
	registry := routerruntime.NewRegistry(nil)
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{
		Config: protected.Config(), ConfigSnapshot: protected,
		AcquireClassification: func() (func(), bool) { return func() {}, true },
	})
	server := &ClassificationAPIServer{config: protected.Config(), runtimeRegistry: registry}
	mux := server.setupRoutes()
	request := json.RawMessage(`{"model":"native/auto","state":"text","questions":{"a":{"type":"choice"}}}`)

	for _, attempt := range []systemone.ForwardRequest{
		{Listener: "public", Method: http.MethodPost, Path: "/v1/systemone", Request: request},
		{Listener: "public", Method: http.MethodPost, Path: "/v1/systemone", Request: request, Authorization: "Bearer native-2"},
		{Listener: "public", Method: http.MethodPost, Path: "/v1/systemone", Request: request, APIKey: "native-1"},
	} {
		// #nosec G117 -- Test-only keys exercise authenticated forwarding.
		body, err := json.Marshal(attempt)
		if err != nil {
			t.Fatal(err)
		}
		response := httptest.NewRecorder()
		mux.ServeHTTP(response, httptest.NewRequest(http.MethodPost, systemone.ForwardPath, strings.NewReader(string(body))))
		if response.Code != http.StatusUnauthorized || !strings.Contains(response.Body.String(), "invalid_api_key") {
			t.Fatalf("forward with %q / %q: %d %s, want 401 invalid_api_key", attempt.Authorization, attempt.APIKey, response.Code, response.Body.String())
		}
	}

	// The forward retained the public generation; the protected one is active
	// by the time the handler reads the model.
	var calls atomic.Int32
	listener, err := systemone.SelectListener(public.Config().Listeners, "public")
	if err != nil {
		t.Fatal(err)
	}
	handler := systemone.Handler(public.Config(), listener, func(context.Context, string, json.RawMessage) (int, []byte, error) {
		calls.Add(1)
		return http.StatusOK, []byte(`{"answers":{}}`), nil
	}, server.forwardClientAuth(public))
	forwarded := httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(string(request)))
	forwarded.Header.Set("Authorization", "Bearer native-1")
	response := httptest.NewRecorder()
	handler(response, forwarded)
	if response.Code != http.StatusConflict || !strings.Contains(response.Body.String(), "configuration_changed") || calls.Load() != 0 {
		t.Fatalf("a reload before admission: %d %s with %d calls, want 409 configuration_changed and none", response.Code, response.Body.String(), calls.Load())
	}
}
