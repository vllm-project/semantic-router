//go:build !windows

package apiserver

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

func TestModelRuntimeInventoryListsPublishedDeploymentsWithoutCredentials(t *testing.T) {
	fake := runtimetest.New(
		runtimetest.Model{ID: "kai"},
		runtimetest.Model{ID: "guard", Device: "cpu", Heads: []runtimetest.Head{{Name: "default", Kind: "sequence", Labels: []string{"benign", "jailbreak"}}}},
	)
	server := httptest.NewServer(fake.Handler())
	defer server.Close()
	endpoint := strings.Replace(server.URL, "http://", "http://operator:secret@", 1)
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{
		"kai":   {Provider: config.ModelRuntimeProvider, Endpoint: endpoint},
		"guard": {Provider: config.ModelRuntimeProvider, Endpoint: endpoint},
	}
	for _, name := range []string{"guard", "kai"} {
		cfg.DecisionRules = append(cfg.DecisionRules, config.DecisionSignalRule{
			Name: "hard_" + name, Deployment: name, Question: config.DecisionQuestion{Type: "noul", Instructions: "Hard?"},
		})
	}
	manager := modelservice.NewManager()
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	if err := manager.Reconcile(cfg); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	for _, name := range []string{"guard", "kai"} {
		if _, err := manager.Published().Card(ctx, name); err != nil {
			t.Fatalf("%s never became ready: %v", name, err)
		}
	}
	previous := modelservice.DefaultManager()
	modelservice.SetDefault(manager)
	t.Cleanup(func() { modelservice.SetDefault(previous) })

	recorder := httptest.NewRecorder()
	(&ClassificationAPIServer{}).handleModelRuntimeInventory(recorder, httptest.NewRequest(http.MethodGet, apiInventoryModelRuntime, nil))
	if recorder.Code != http.StatusOK {
		t.Fatalf("status %d: %s", recorder.Code, recorder.Body.String())
	}
	if body := recorder.Body.String(); strings.Contains(body, "secret") || strings.Contains(body, "operator") {
		t.Fatalf("the inventory leaked endpoint credentials: %s", body)
	}
	var response modelRuntimeInventoryResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response.Count != 2 || len(response.Deployments) != 2 {
		t.Fatalf("inventory = %+v", response)
	}
	guard, kai := response.Deployments[0], response.Deployments[1]
	if guard.Name != "guard" || guard.Managed || guard.ServedName != "guard" || guard.Endpoint != server.URL || !guard.Ready || guard.Restarts != 0 {
		t.Fatalf("guard = %+v", guard)
	}
	wantHeads := []modelRuntimeHead{{Name: "default", Kind: "sequence", Labels: []string{"benign", "jailbreak"}}}
	if guard.Family != "task_heads" || !reflect.DeepEqual(guard.Surfaces, []string{"classify"}) || !reflect.DeepEqual(guard.Heads, wantHeads) || guard.Device != "cpu" || guard.Profile != "exact" {
		t.Fatalf("guard card = %+v", guard)
	}
	if kai.Family != "decision2" || !reflect.DeepEqual(kai.Surfaces, []string{"decisions"}) || len(kai.Heads) != 0 {
		t.Fatalf("kai card = %+v", kai)
	}
}

func TestModelRuntimeInventoryIsEmptyBeforeTheRuntimeStarts(t *testing.T) {
	previous := modelservice.DefaultManager()
	modelservice.SetDefault(nil)
	t.Cleanup(func() { modelservice.SetDefault(previous) })

	recorder := httptest.NewRecorder()
	(&ClassificationAPIServer{}).handleModelRuntimeInventory(recorder, httptest.NewRequest(http.MethodGet, apiInventoryModelRuntime, nil))
	var response map[string]any
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if deployments, ok := response["deployments"].([]any); recorder.Code != http.StatusOK || !ok || len(deployments) != 0 {
		t.Fatalf("status %d: %s", recorder.Code, recorder.Body.String())
	}
}

func TestInventoryEndpointKeepsOnlyTheTransport(t *testing.T) {
	for endpoint, want := range map[string]string{
		"unix:///run/vllm-sr/cpu.sock":                    "unix",
		"https://operator:secret@runtime:8443/v1?token=x": "https://runtime:8443",
		"http://127.0.0.1:8100":                           "http://127.0.0.1:8100",
		"":                                                "",
	} {
		if got := inventoryEndpoint(endpoint); got != want {
			t.Errorf("inventoryEndpoint(%q) = %q, want %q", endpoint, got, want)
		}
	}
}

func TestModelRuntimeInventoryPreservesLogicalReplicaStatus(t *testing.T) {
	inventory := modelRuntimeInventory([]modelservice.DeploymentStatus{{Name: "primary", Ready: true, State: "degraded", DesiredReplicas: 2, ReadyReplicas: 1, Replicas: []modelservice.ReplicaStatus{{ID: "r-opaque", Managed: true, Device: "rocm:0", Ready: true, Inflight: 2, EstimatedWork: 4096}, {ID: "r-other", State: "backoff"}}}})
	if inventory.Count != 1 || inventory.Deployments[0].DesiredReplicas != 2 || inventory.Deployments[0].ReadyReplicas != 1 || inventory.Deployments[0].Replicas[0].Inflight != 2 {
		t.Fatalf("pool inventory lost: %+v", inventory)
	}
	encoded, _ := json.Marshal(inventory)
	if strings.Contains(string(encoded), "http://") || strings.Contains(string(encoded), "unix://") {
		t.Fatalf("worker location leaked: %s", encoded)
	}
}
