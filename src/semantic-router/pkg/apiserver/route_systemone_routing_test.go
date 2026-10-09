//go:build !windows

package apiserver

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
)

type diagnosticNativeRouter struct {
	t     *testing.T
	held  *int
	calls int
}

func (r *diagnosticNativeRouter) RouteSystemOne(_ context.Context, model string, body json.RawMessage, _ systemone.Invoke) (int, []byte, error) {
	r.calls++
	if *r.held != 1 || model != "native-auto" || !strings.Contains(string(body), `"model":"native-auto"`) || !strings.Contains(string(body), `"questions":{"z":{"type":"noul"},"a":{"type":"choice"}}`) {
		r.t.Fatal("operator test changed the task or lost its active generation")
	}
	return http.StatusOK, []byte(`{"model":"internal","answers":{},"routing":{"recipe":"native"}}`), nil
}

func TestSystemOneRouteDiagnosticsUseOperatorPermissionAndActiveGeneration(t *testing.T) {
	t.Setenv("NATIVE_ROUTE_OPERATOR", "operator-test-key")
	t.Setenv("NATIVE_ROUTE_VIEWER", "viewer-test-key")
	cfg, err := config.ParseYAMLBytes([]byte(`version: v0.3
listeners:
  - name: public
    port: 8899
    api_keys: [public-test-key]
    systemone: {models: [public-native]}
providers:
  models:
    - name: fast
      api_format: systemone
      backend_refs: [{provider: systemone-compatible, base_url: https://example.invalid/v1}]
routing: {}
entrypoints:
  - {api: systemone, model_names: [native-auto, public-native], recipe: native}
recipes:
  - name: native
    routing:
      signals:
        keywords: [{name: greeting, operator: OR, keywords: [hello]}]
      decisions:
        - name: classify
          rules: {type: keyword, name: greeting}
          modelRefs: [{model: fast}]
          algorithm: &simple
            type: cascade
            budget: {deadline: 2s, max_calls: 1}
            quality:
              type: uncalibrated
              acceptance:
                rules: [{question_type: choice, field: top_probability, predicate: {gte: 0}}]
            stages: [{name: fast, kind: native, model: fast}]
        - name: complex
          rules: {}
          modelRefs: [{model: fast}]
          algorithm:
            <<: *simple
            budget: {deadline: 4s, max_calls: 2}
`))
	if err != nil {
		t.Fatal(err)
	}
	// The recipe is active, but native-auto has no public listener grant.
	// Only managed classify.invoke authorizes this operator diagnostic.
	cfg.ManagementAPI = config.ManagementAPIConfig{Auth: config.ManagementAPIAuthConfig{
		Mode: config.ManagementAuthModeBearer,
		Tokens: []config.ManagementAPITokenRef{
			{Env: "NATIVE_ROUTE_OPERATOR", Role: "operator"},
			{Env: "NATIVE_ROUTE_VIEWER", Role: "reader"},
		},
		Roles: map[string][]string{"operator": {string(PermClassifyInvoke), string(PermConfigRead)}, "reader": {string(PermConfigRead)}},
	}}
	snapshot, err := configsnapshot.NewManager(configsnapshot.Options{}).Install(context.Background(), configsnapshot.Update{Config: cfg})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = snapshot.Release(context.Background()) })
	held := 0
	router := &diagnosticNativeRouter{t: t, held: &held}
	registry := routerruntime.NewRegistry(nil)
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{
		Config: cfg, ConfigSnapshot: snapshot, NativeRouter: router,
		AcquireClassification: func() (func(), bool) { held++; return func() { held-- }, true },
	})
	server := &ClassificationAPIServer{config: cfg, runtimeRegistry: registry}
	mux := server.setupRoutes()
	body := `{"model":"native-auto","request":{"model":"ignored","state":"hello","questions":{"z":{"type":"noul"},"a":{"type":"choice"}}}}`
	for _, tc := range []struct {
		method, key string
		status      int
	}{
		{http.MethodGet, "", http.StatusUnauthorized},
		{http.MethodGet, "viewer-test-key", http.StatusOK},
		{http.MethodPost, "viewer-test-key", http.StatusForbidden},
		{http.MethodPost, "operator-test-key", http.StatusOK},
	} {
		r := httptest.NewRequest(tc.method, systemOneRoutingDiagnosticPath, strings.NewReader(body))
		r.Header.Set("Authorization", "Bearer "+tc.key)
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, r)
		if w.Code != tc.status || held != 0 {
			t.Fatalf("%s status=%d held=%d body=%s", tc.method, w.Code, held, w.Body.String())
		}
		if tc.status == http.StatusOK && (!strings.Contains(w.Body.String(), "native-auto") || !strings.Contains(w.Body.String(), "native")) {
			t.Fatal("operator diagnostic lost active route identity")
		}
		if tc.status == http.StatusOK && tc.method == http.MethodGet {
			var routes SystemOneRoutes
			if json.Unmarshal(w.Body.Bytes(), &routes) != nil || len(routes.Routes) != 2 {
				t.Fatal("operator diagnostic lost route discovery")
			}
			for _, route := range routes.Routes {
				if route.ExecutionTimeoutMS != 4000 {
					t.Fatalf("discovery did not report the maximum algorithm deadline: %+v", route)
				}
			}
		}
	}
	if router.calls != 1 {
		t.Fatalf("unauthorized request or discovery reached models: calls=%d", router.calls)
	}
	request := httptest.NewRequest(http.MethodPost, systemOneRoutingDiagnosticPath, strings.NewReader(strings.Replace(body, "native-auto", "unknown", 1)))
	request.Header.Set("Authorization", "Bearer operator-test-key")
	response := httptest.NewRecorder()
	mux.ServeHTTP(response, request)
	if response.Code != http.StatusNotFound || router.calls != 1 || held != 0 {
		t.Fatal("unknown route was executed or leaked its lease")
	}
}
