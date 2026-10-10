package modelservice

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestHostRuntimeGenerationLeases(t *testing.T) {
	var mu sync.Mutex
	starts, stops := 0, 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer private-test-token" {
			t.Errorf("missing private host credential")
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodPost:
			var body struct {
				Models []modelEntry `json:"models"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			if len(body.Models) != 1 || body.Models[0].Device != "mps" || body.Models[0].Name != "shared" {
				t.Errorf("unexpected host model plan: %+v", body.Models)
			}
			mu.Lock()
			starts++
			mu.Unlock()
			_, _ = w.Write([]byte(`{"port":12345}`))
		case http.MethodDelete:
			mu.Lock()
			stops++
			mu.Unlock()
			_, _ = w.Write([]byte(`{"stopped":true}`))
		default:
			if !strings.HasSuffix(r.URL.Path, "/health") {
				t.Errorf("unexpected runtime call %s", r.URL.Path)
			}
			_, _ = w.Write([]byte(`{"status":"starting","api_version":"2.0"}`))
		}
	}))
	defer server.Close()
	t.Setenv(HostRuntimeEndpointEnv, server.URL)
	t.Setenv(HostRuntimeTokenEnv, "private-test-token")
	t.Setenv(RuntimeCommandEnv, "must-not-execute-on-linux")
	manager := NewManager()
	deployments := map[string]config.ModelDeployment{"shared": {
		Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/fixture", Device: "cpu",
	}}
	first, err := manager.AcquireDeployments(deployments)
	if err != nil {
		t.Fatal(err)
	}
	second, err := manager.AcquireDeployments(deployments)
	if err != nil {
		t.Fatal(err)
	}
	if err := first.Close(); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	if starts != 1 || stops != 0 {
		t.Errorf("reload did not retain host process: starts=%d stops=%d", starts, stops)
	}
	mu.Unlock()
	if err := second.Close(); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if stops != 1 {
		t.Errorf("last generation did not release host process: stops=%d", stops)
	}
	if deployments["shared"].Device != "cpu" {
		t.Error("host placement mutated authored deployment")
	}
}

func TestHostPlacementPreservesAttachedDeployments(t *testing.T) {
	manager := &Manager{host: &hostRuntime{endpoint: "http://host.docker.internal:1"}, cores: 4}
	plans := manager.processPlans(map[string]config.ModelDeployment{
		"external": {Provider: config.ModelRuntimeProvider, Endpoint: "http://external:8000", ServedName: "external-model"},
	})
	if len(plans) != 1 || plans[0].endpoint != "http://external:8000" || len(plans[0].models) != 0 {
		t.Fatalf("external runtime was relocated onto the host: %+v", plans)
	}
}

func TestHostPlacementPreservesAuthoredReplicas(t *testing.T) {
	manager := &Manager{host: &hostRuntime{}, cores: 4}
	replicas := []config.ModelReplica{{Device: "cpu"}, {Endpoint: "http://external:8000"}}
	plans := manager.processPlans(map[string]config.ModelDeployment{
		"pool": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/fixture", Replicas: replicas},
	})
	managed, attached := 0, 0
	for _, plan := range plans {
		if plan.endpoint != "" {
			attached++
		} else {
			managed++
			if plan.models[0].Device != "mps" {
				t.Fatal("managed replica did not select MPS")
			}
		}
	}
	if managed != 1 || attached != 1 || replicas[0].Device != "cpu" {
		t.Fatal("host placement changed replica ownership or authored configuration")
	}
}
