package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestEngineFrontendNativeInferenceNeedsNeitherRouterNorUpstream(t *testing.T) {
	fake := runtimetest.New(runtimetest.Model{ID: "served", Repo: "example/vela"})
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/systemone" {
			r.URL.Path = "/v1/decisions"
		}
		fake.Handler().ServeHTTP(w, r)
	}))
	defer worker.Close()
	manager := modelservice.NewManager()
	previous := modelservice.DefaultManager()
	modelservice.SetDefault(manager)
	t.Cleanup(func() { modelservice.SetDefault(previous); _ = manager.Shutdown(context.Background()) })
	disabled := false
	cfg := &config.RouterConfig{RouterOptions: config.RouterOptions{RouterEnabled: &disabled}}
	cfg.DecisionModel = "primary"
	cfg.ModelDeployments = map[string]config.ModelDeployment{"primary": {Provider: config.ModelRuntimeProvider, Artifact: "example/vela", Endpoint: worker.URL, ServedName: "served"}}
	cfg.Listeners = []config.Listener{{Name: "native", Port: 8899, APIKeys: []string{"key"}, SystemOne: &config.ListenerSystemOne{Models: []string{"example/vela"}}}}
	if err := manager.Reconcile(cfg); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if _, err := manager.Published().Card(ctx, "primary"); err != nil {
		t.Fatal(err)
	}
	if result, callErr := manager.Published().SystemOne(ctx, "primary", json.RawMessage(`{"state":"test","questions":{"q":{"type":"noul","instructions":"Test?"}}}`)); callErr != nil {
		t.Fatalf("retained manager native request: %+v %v", result, callErr)
	}
	registry := routerruntime.NewRegistry(cfg)
	server, err := extproc.NewServer(t.TempDir()+"/config.yaml", 0, false, "", registry, extproc.WithGatewayMode(config.GatewayStandalone))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = server.Shutdown(context.Background()) })
	if server.GetRouter() != nil {
		t.Fatal("Engine constructed Router")
	}
	handler, err := gateway.NewHandler(gateway.Options{Listener: "native", Serving: nativeServing{pin: server.Pin}})
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		path, token, model string
		status             int
	}{
		{"/v1/systemone", "key", "example/vela", http.StatusOK},
		{"/v1/decisions", "key", "example/vela", http.StatusOK},
		{"/v1/systemone", "", "example/vela", http.StatusUnauthorized},
		{"/v1/systemone", "key", "undeclared", http.StatusForbidden},
		{"/v1/chat/completions", "key", "example/vela", http.StatusNotFound},
	} {
		request := httptest.NewRequest(http.MethodPost, tc.path, strings.NewReader(`{"model":"`+tc.model+`","state":"A question","questions":{"q":{"type":"noul","instructions":"Is this a question?"}}}`))
		if tc.token != "" {
			request.Header.Set("Authorization", "Bearer "+tc.token)
		}
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != tc.status {
			t.Fatalf("%s %s: %d %s", tc.path, tc.model, response.Code, response.Body.String())
		}
	}
}
