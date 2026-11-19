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

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/clientauth"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
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

type forwardClientAuthRouter struct{ calls, backendCalls int }

func (r *forwardClientAuthRouter) RouteSystemOne(ctx context.Context, _ string, body json.RawMessage, invoke systemone.Invoke) (int, []byte, error) {
	r.calls++
	return invoke(ctx, "kai", body)
}

func (*forwardClientAuthRouter) Close(context.Context) error { return nil }

func (r *forwardClientAuthRouter) SystemOne(context.Context, string, json.RawMessage) (modelservice.SystemOneResult, error) {
	r.backendCalls++
	return modelservice.SystemOneResult{Status: http.StatusOK, Body: json.RawMessage(`{"answers":{}}`)}, nil
}

// The serving pointer changes before the management registry is published.
// Exercise the real forward handler inside that handoff, while the old
// generation can still be acquired, rather than after publication completes.
func TestSystemOneForwardRefusesChangedPolicyDuringHandoff(t *testing.T) {
	t.Setenv("NATIVE_CLIENT_KEY", "native-1")
	cfg, err := config.ParseYAMLBytes([]byte(forwardClientAuthConfig))
	if err != nil {
		t.Fatal(err)
	}
	cfg.Entrypoints[1].ClientAuth = &config.EntrypointClientAuth{Type: config.ClientAuthBearerAPIKey, APIKeyEnvs: []string{"NATIVE_CLIENT_KEY"}}
	router := &forwardClientAuthRouter{}
	snapshot, err := configsnapshot.NewManager(configsnapshot.Options{Parts: []configsnapshot.PartBuilder{{
		Component: configsnapshot.ComponentModelService,
		Build: func(context.Context, *configsnapshot.Snapshot, configsnapshot.Part) (configsnapshot.Part, error) {
			return router, nil
		},
	}}}).Install(context.Background(), configsnapshot.Update{Config: cfg})
	if err != nil {
		t.Fatal(err)
	}
	service := extproc.NewRouterServiceForSnapshot(nil, snapshot)
	t.Cleanup(func() { _ = service.Close() })
	lease, err := service.Pin()
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	registry := routerruntime.NewRegistry(nil)
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{
		Config: cfg, ConfigSnapshot: snapshot, NativeRouter: router,
		ActiveClientAuth:      lease.ActiveClientAuth,
		AcquireClassification: func() (func(), bool) { return func() {}, true },
	})
	mux := (&ClassificationAPIServer{config: cfg, runtimeRegistry: registry}).setupRoutes()
	send := func() *httptest.ResponseRecorder {
		body := `{"listener":"public","method":"POST","path":"/v1/systemone","authorization":"Bearer native-1","request":{"model":"native/auto","state":"text","questions":{"a":{"type":"choice"}}}}`
		response := httptest.NewRecorder()
		mux.ServeHTTP(response, httptest.NewRequest(http.MethodPost, systemone.ForwardPath, strings.NewReader(body)))
		return response
	}
	if response := send(); response.Code != http.StatusOK || router.calls != 1 || router.backendCalls != 1 {
		t.Fatalf("before reload: %d %s with %d routing / %d backend calls, want 200 and one of each", response.Code, response.Body.String(), router.calls, router.backendCalls)
	}
	router.calls, router.backendCalls = 0, 0

	// Swap to a generation without client keys. The callback is a deterministic
	// barrier after the serving pointer changes and before registry publication.
	err = service.Swap(nil, func(extproc.AcquireFunc) {
		if registry.ConfigSnapshot() != snapshot {
			t.Error("the test missed the publication handoff")
		}
		if got := clientauth.Admit(snapshot.ClientAuth(), lease.ActiveClientAuth(), config.SystemOneAPI, "native/auto", "Bearer native-1"); got != clientauth.Superseded {
			t.Error("the serving policy did not change during the handoff")
		}
		response := send()
		if response.Code != http.StatusConflict || !strings.Contains(response.Body.String(), "configuration_changed") || router.calls != 0 || router.backendCalls != 0 {
			t.Errorf("handoff: %d %s with %d routing / %d backend calls, want 409 configuration_changed and no calls", response.Code, response.Body.String(), router.calls, router.backendCalls)
		}
	})
	if err != nil {
		t.Fatal(err)
	}
}
