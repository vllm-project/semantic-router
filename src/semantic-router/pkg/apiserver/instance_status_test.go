//go:build !windows

package apiserver

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestInstanceStatusReportsOnlyActiveFrontendCapability(t *testing.T) {
	disabled := false
	cfg := &config.RouterConfig{RouterOptions: config.RouterOptions{RouterEnabled: &disabled}}
	cfg.DecisionModel = "primary"
	cfg.ModelDeployments = map[string]config.ModelDeployment{"primary": {Provider: config.ModelRuntimeProvider, Artifact: "example/vela"}}
	registry := routerruntime.NewRegistry(cfg)
	server := &ClassificationAPIServer{runtimeRegistry: registry}
	read := func() InstanceStatus {
		response := httptest.NewRecorder()
		server.handleInstanceStatus(response, httptest.NewRequest(http.MethodGet, "/api/v1/instance", nil))
		var result InstanceStatus
		if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
			t.Fatal(err)
		}
		return result
	}
	if state := read(); state.ObservedMode != "unknown" {
		t.Fatal("parsed config was claimed active")
	}
	manager := configsnapshot.NewManager(configsnapshot.Options{})
	snapshot, err := manager.Install(context.Background(), configsnapshot.Update{Config: cfg})
	if err != nil {
		t.Fatal(err)
	}
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{Config: cfg, ConfigSnapshot: snapshot})
	if state := read(); state.ObservedMode != "engine" || state.ActiveDeployment != "primary" || state.Model != "example/vela" {
		t.Fatalf("state=%+v", state)
	}
	response := httptest.NewRecorder()
	server.handleHealth(response, httptest.NewRequest(http.MethodGet, "/health", nil))
	var health healthResponse
	if err := json.Unmarshal(response.Body.Bytes(), &health); err != nil {
		t.Fatal(err)
	}
	if health.ServingMode != "engine" || health.Status != "healthy" {
		t.Fatalf("health=%+v", health)
	}
}
