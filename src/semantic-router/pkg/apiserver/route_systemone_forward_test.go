//go:build !windows

package apiserver

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
)

type forwardTestModels struct {
	held   *atomic.Int32
	calls  atomic.Int32
	during func()
	t      *testing.T
}

func (*forwardTestModels) Close(context.Context) error { return nil }

func (m *forwardTestModels) SystemOne(_ context.Context, deployment string, body json.RawMessage) (modelservice.SystemOneResult, error) {
	if m.held.Load() != 1 {
		m.t.Fatal("inference did not retain the authorized generation")
	}
	m.calls.Add(1)
	if m.during != nil {
		m.during()
	}
	if deployment != "primary" && deployment != "second" {
		m.t.Fatalf("unexpected deployment %q", deployment)
	}
	if !strings.Contains(string(body), `"questions":{"z":{"type":"noul"},"a":{"type":"choice"}}`) {
		m.t.Fatalf("native question order or types changed: %s", body)
	}
	return modelservice.SystemOneResult{Status: 200, Body: json.RawMessage(`{"model":"internal","answers":{}}`)}, nil
}

func TestSystemOneForwardUsesActiveListenerAndRetainsGeneration(t *testing.T) {
	for _, routing := range []bool{false, true} {
		name := "engine"
		if routing {
			name = "router"
		}
		t.Run(name, func(t *testing.T) {
			var held atomic.Int32
			models := &forwardTestModels{t: t, held: &held}
			registry := routerruntime.NewRegistry(nil)
			makeSnapshot := func(key string, grants []string) *configsnapshot.Snapshot {
				cfg := &config.RouterConfig{RouterOptions: config.RouterOptions{RouterEnabled: &routing}}
				cfg.Listeners = []config.Listener{
					{Name: "public", APIKeys: []string{key}, SystemOne: &config.ListenerSystemOne{Models: grants}},
					{Name: "private", APIKeys: []string{"private-key"}, SystemOne: &config.ListenerSystemOne{Models: []string{"judge-next"}}},
				}
				cfg.ModelDeployments = map[string]config.ModelDeployment{
					"primary": {Provider: config.ModelRuntimeProvider, Artifact: "example/judge", PublicName: "judge"},
					"second":  {Provider: config.ModelRuntimeProvider, Artifact: "example/second", PublicName: "judge-next"},
				}
				manager := configsnapshot.NewManager(configsnapshot.Options{Parts: []configsnapshot.PartBuilder{{
					Component: configsnapshot.ComponentModelService,
					Build: func(context.Context, *configsnapshot.Snapshot, configsnapshot.Part) (configsnapshot.Part, error) {
						return models, nil
					},
				}}})
				snapshot, err := manager.Install(context.Background(), configsnapshot.Update{Config: cfg})
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { _ = snapshot.Release(context.Background()) })
				return snapshot
			}
			active := makeSnapshot("active-key", []string{"judge"})
			pending := makeSnapshot("pending-key", []string{"judge", "judge-next"})
			publish := func(snapshot *configsnapshot.Snapshot) {
				registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{Config: snapshot.Config(), ConfigSnapshot: snapshot, AcquireClassification: func() (func(), bool) {
					held.Add(1)
					return func() { held.Add(-1) }, true
				}})
			}
			publish(active)
			configPath := filepath.Join(t.TempDir(), "config.yaml")
			data, err := json.Marshal(config.CanonicalConfigFromRouterConfig(pending.Config()))
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(configPath, data, 0o600); err != nil {
				t.Fatal(err)
			}
			server := &ClassificationAPIServer{config: pending.Config(), configPath: configPath, runtimeRegistry: registry}
			mux := server.setupRoutes()
			invoke := func(key, model, listener string, discovery bool) *httptest.ResponseRecorder {
				request := systemone.ForwardRequest{
					Listener: listener, Method: http.MethodPost, Path: "/v1/systemone", Authorization: "Bearer " + key,
					BackendRequest: true,
					Request:        json.RawMessage(`{"model":"` + model + `","state":"text","questions":{"z":{"type":"noul"},"a":{"type":"choice"}}}`),
				}
				if discovery {
					request.Method, request.Path, request.Request = http.MethodGet, "/v1/systemone/models", nil
				}
				// #nosec G117 -- Test-only keys exercise authenticated forwarding; no real credentials are logged or persisted.
				body, err := json.Marshal(request)
				if err != nil {
					t.Fatal(err)
				}
				response := httptest.NewRecorder()
				mux.ServeHTTP(response, httptest.NewRequest(http.MethodPost, systemone.ForwardPath, strings.NewReader(string(body))))
				if held.Load() != 0 {
					t.Fatal("forward leaked its generation lease")
				}
				return response
			}
			for _, test := range []struct {
				key, model, listener string
				discovery            bool
				status               int
			}{
				{"pending-key", "judge", "public", false, 401},
				{"pending-key", "", "public", true, 401},
				{"active-key", "judge-next", "public", false, 403},
				{"active-key", "judge-next", "private", false, 401},
				{"active-key", "judge", "", false, 404},
			} {
				response := invoke(test.key, test.model, test.listener, test.discovery)
				if response.Code != test.status {
					t.Fatalf("%+v => %d %s", test, response.Code, response.Body.String())
				}
			}
			if models.calls.Load() != 0 {
				t.Fatal("pending or another listener's grant reached inference")
			}
			if response := invoke("active-key", "", "public", true); response.Code != 200 || strings.Contains(response.Body.String(), "judge-next") {
				t.Fatalf("pending discovery grant exposed: %d %s", response.Code, response.Body.String())
			}
			models.during = func() { publish(pending) }
			if response := invoke("active-key", "judge", "public", false); response.Code != 200 || !strings.Contains(response.Body.String(), `"model":"judge"`) {
				t.Fatalf("active inference failed across activation: %d %s", response.Code, response.Body.String())
			}
			models.during = nil
			if response := invoke("active-key", "judge", "public", false); response.Code != 401 {
				t.Fatal("previous credentials remained authorized after activation")
			}
			if response := invoke("pending-key", "judge-next", "public", false); response.Code != 200 {
				t.Fatalf("activated grant unavailable: %d %s", response.Code, response.Body.String())
			}
		})
	}
}

func TestSystemOneForwardRequiresManagementInvokePermission(t *testing.T) {
	t.Setenv("SYSTEMONE_FORWARD_INSPECTOR", "test-inspector")
	server := testManagementAPIServer(t, config.ManagementAPIConfig{Auth: config.ManagementAPIAuthConfig{Mode: config.ManagementAuthModeBearer, Tokens: []config.ManagementAPITokenRef{{Env: "SYSTEMONE_FORWARD_INSPECTOR", Role: "inspector"}}, Roles: map[string][]string{"inspector": {"config.read"}}}})
	for _, token := range []string{"", "test-inspector"} {
		request := httptest.NewRequest(http.MethodPost, systemone.ForwardPath, strings.NewReader(`{"method":"GET","path":"/v1/systemone/models"}`))
		if token != "" {
			request.Header.Set("Authorization", "Bearer "+token)
		}
		response := httptest.NewRecorder()
		server.setupRoutes().ServeHTTP(response, request)
		want := http.StatusUnauthorized
		if token != "" {
			want = http.StatusForbidden
		}
		if response.Code != want {
			t.Fatalf("status=%d, want %d", response.Code, want)
		}
	}
}

func TestSystemOneForwardRejectsUnknownTargetsAndMissingActiveGeneration(t *testing.T) {
	mux := (&ClassificationAPIServer{}).setupRoutes()
	for _, test := range []struct {
		body string
		want int
	}{
		{`{"method":"POST","path":"/v1/systemone","url":"http://untrusted"}`, 400},
		{`{"method":"POST","path":"/v1/systemone","deployment":"private"}`, 400},
		{`{"method":"POST","path":"/api/v1/config"}`, 400},
		{`{"method":"POST","path":"/v1/systemone/models"}`, 400},
		{strings.Repeat("x", systemOneForwardRequestLimit+1), 413},
		{`{"method":"GET","path":"/v1/systemone/models"}`, 503},
	} {
		response := httptest.NewRecorder()
		mux.ServeHTTP(response, httptest.NewRequest(http.MethodPost, systemone.ForwardPath, strings.NewReader(test.body)))
		if response.Code != test.want {
			t.Fatalf("status=%d, want %d: %s", response.Code, test.want, response.Body.String())
		}
	}
}
