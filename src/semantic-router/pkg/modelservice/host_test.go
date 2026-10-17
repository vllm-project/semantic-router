package modelservice

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

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
		Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/fixture", Device: "auto",
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

func TestHostRuntimeConcurrentReacquisitionKeepsNewLease(t *testing.T) {
	var mu sync.Mutex
	owners := make(map[string]struct{})
	deleting := make(chan struct{})
	resume := make(chan struct{})
	var firstDelete, unblock sync.Once
	unblockDelete := func() { unblock.Do(func() { close(resume) }) }
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodPost || r.Method == http.MethodDelete {
			var body struct {
				Lease string `json:"lease"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			if body.Lease == "" {
				t.Error("host control request omitted its owning lease")
			}
			if r.Method == http.MethodDelete {
				firstDelete.Do(func() {
					close(deleting)
					<-resume
				})
			}
			mu.Lock()
			if r.Method == http.MethodPost {
				owners[body.Lease] = struct{}{}
			} else {
				delete(owners, body.Lease)
			}
			mu.Unlock()
			_, _ = w.Write([]byte(`{"port":12345}`))
			return
		}
		_, _ = w.Write([]byte(`{"status":"starting","api_version":"2.0"}`))
	}))
	defer server.Close()
	defer unblockDelete()
	t.Setenv(HostRuntimeEndpointEnv, server.URL)
	t.Setenv(HostRuntimeTokenEnv, "private-test-token")
	t.Setenv(RuntimeCommandEnv, "must-not-execute-on-linux")
	manager := NewManager()
	deployments := map[string]config.ModelDeployment{"shared": {
		Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/fixture", Device: "auto",
	}}
	first, err := manager.AcquireDeployments(deployments)
	if err != nil {
		t.Fatal(err)
	}
	retired := make(chan error, 1)
	go func() { retired <- first.Close() }()
	select {
	case <-deleting:
	case <-time.After(5 * time.Second):
		t.Fatal("retiring group did not reach host DELETE")
	}
	second, err := manager.AcquireDeployments(deployments)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		unblockDelete()
		if err := second.Close(); err != nil {
			t.Error(err)
		}
	}()
	mu.Lock()
	ownerCount := len(owners)
	mu.Unlock()
	if ownerCount != 2 {
		t.Errorf("reacquisition must have a distinct owning lease; got %d", ownerCount)
	}
	unblockDelete()
	select {
	case err := <-retired:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("retiring group did not finish after DELETE resumed")
	}
	mu.Lock()
	ownerCount = len(owners)
	mu.Unlock()
	if ownerCount != 1 {
		t.Fatalf("retired DELETE removed the newly acquired process lease; got %d", ownerCount)
	}
	if err := second.Close(); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(owners) != 0 {
		t.Error("final owning lease did not release the host process")
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

func TestExplicitCPUStaysWithTheContainerSupervisor(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("explicit CPU deployment contacted the Apple host")
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv(fakeRuntimeEnv, "1")
	t.Setenv(RuntimeCommandEnv, binary)
	t.Setenv(RuntimeDirEnv, filepath.Join(t.TempDir(), "run"))
	t.Setenv(HostRuntimeEndpointEnv, server.URL)
	t.Setenv(HostRuntimeTokenEnv, "private-test-token")
	manager := NewManager()
	defer func() { _ = manager.Shutdown(context.Background()) }()
	lease, err := manager.AcquireDeployments(map[string]config.ModelDeployment{
		"cpu": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/fixture", Device: "cpu"},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = lease.Close() }()
	if len(lease.groups) != 1 || lease.groups[0].host != nil || lease.groups[0].supervisor == nil {
		t.Fatal("explicit CPU placement did not retain the local managed supervisor")
	}
	waitReady(t, lease, "cpu")
}

func TestHostPlacementPreservesAuthoredReplicas(t *testing.T) {
	manager := &Manager{host: &hostRuntime{}, cores: 4}
	replicas := []config.ModelReplica{{Device: "cpu"}, {Device: "auto"}, {Endpoint: "http://external:8000"}}
	plans := manager.processPlans(map[string]config.ModelDeployment{
		"pool": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/fixture", Replicas: replicas},
	})
	cpu, gpu, attached := 0, 0, 0
	for _, plan := range plans {
		if plan.endpoint != "" {
			attached++
		} else {
			switch plan.models[0].Device {
			case "cpu":
				cpu++
			case "mps":
				gpu++
			default:
				t.Fatal("managed replica lost its authored placement")
			}
		}
	}
	if cpu != 1 || gpu != 1 || attached != 1 || replicas[0].Device != "cpu" || replicas[1].Device != "auto" {
		t.Fatal("host placement changed replica ownership or authored configuration")
	}
}
