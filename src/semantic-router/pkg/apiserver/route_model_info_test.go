//go:build !windows

package apiserver

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func requireModelInfo(t *testing.T, models []ModelInfo, name string) ModelInfo {
	t.Helper()

	for _, model := range models {
		if model.Name == name {
			return model
		}
	}

	t.Fatalf("expected %s entry, got %+v", name, models)
	return ModelInfo{}
}

func TestLoadModelsRuntimeStateUsesReplicaLocalSnapshot(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "router-config.yaml")
	cfg := &config.RouterConfig{StartupStatus: config.StartupStatusConfig{StoreBackend: "file"}}
	local := routerruntime.NewRegistry(cfg)
	other := routerruntime.NewRegistry(cfg)
	localWriter := local.StartupStatusWriter(startupstatus.NewFileWriter(configPath))
	otherWriter := other.StartupStatusWriter(startupstatus.NewFileWriter(configPath))

	localState := startupstatus.State{
		Phase:         "initializing_models",
		PendingModels: []string{"local-model"},
		TotalModels:   2,
		ReadyModels:   1,
	}
	if err := localWriter.Write(localState); err != nil {
		t.Fatalf("write local runtime state: %v", err)
	}
	if err := otherWriter.Write(startupstatus.State{
		Phase:       "ready",
		Ready:       true,
		TotalModels: 99,
		ReadyModels: 99,
	}); err != nil {
		t.Fatalf("write shared runtime state: %v", err)
	}

	apiServer := &ClassificationAPIServer{
		configPath:      configPath,
		runtimeRegistry: local,
	}
	state := apiServer.loadModelsRuntimeState()
	if state == nil {
		t.Fatal("expected replica-local runtime state")
	}
	if state.Phase != localState.Phase || state.Ready != localState.Ready ||
		state.ReadyModels != localState.ReadyModels || state.TotalModels != localState.TotalModels {
		t.Fatalf("loaded another replica's runtime state: %+v", state)
	}

	unobserved := &ClassificationAPIServer{
		configPath:      configPath,
		runtimeRegistry: routerruntime.NewRegistry(cfg),
	}
	if state := unobserved.loadModelsRuntimeState(); state != nil {
		t.Fatalf("unobserved replica loaded shared runtime state: %+v", state)
	}
}

func TestModelsInventoryHTTPKeepsReplicaStatusWhenStartupFileIsShared(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "router-config.yaml")
	cfg := &config.RouterConfig{StartupStatus: config.StartupStatusConfig{StoreBackend: "file"}}
	starting := routerruntime.NewRegistry(cfg)
	ready := routerruntime.NewRegistry(cfg)

	startingServer := &ClassificationAPIServer{config: cfg, configPath: configPath, runtimeRegistry: starting}
	readyServer := &ClassificationAPIServer{config: cfg, configPath: configPath, runtimeRegistry: ready}
	serveInventory := func(api *ClassificationAPIServer) *httptest.Server {
		mux := http.NewServeMux()
		mux.HandleFunc("/api/v1/inventory/models", api.handleModelsInfo)
		server := httptest.NewServer(mux)
		t.Cleanup(server.Close)
		return server
	}
	startingHTTP := serveInventory(startingServer)
	readyHTTP := serveInventory(readyServer)

	startingWriter := starting.StartupStatusWriter(startupstatus.NewFileWriter(configPath))
	readyWriter := ready.StartupStatusWriter(startupstatus.NewFileWriter(configPath))
	if err := startingWriter.Write(startupstatus.State{Phase: "initializing_models", Ready: false, TotalModels: 2, ReadyModels: 1}); err != nil {
		t.Fatal(err)
	}
	if err := readyWriter.Write(startupstatus.State{Phase: "ready", Ready: true, TotalModels: 3, ReadyModels: 3}); err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		name   string
		server *httptest.Server
		phase  string
		ready  bool
		loaded int
		total  int
	}{
		{name: "starting replica", server: startingHTTP, phase: "initializing_models", ready: false, loaded: 1, total: 2},
		{name: "ready replica", server: readyHTTP, phase: "ready", ready: true, loaded: 3, total: 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			response, err := tc.server.Client().Get(tc.server.URL + "/api/v1/inventory/models")
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			if response.StatusCode != http.StatusOK {
				t.Fatalf("inventory status = %d, want 200", response.StatusCode)
			}
			var body ModelsInfoResponse
			if err := json.NewDecoder(response.Body).Decode(&body); err != nil {
				t.Fatal(err)
			}
			if body.Summary.Phase != tc.phase || body.Summary.Ready != tc.ready ||
				body.Summary.LoadedModels != tc.loaded || body.Summary.TotalModels != tc.total {
				t.Fatalf("inventory exposed another replica's startup state: %+v", body.Summary)
			}
		})
	}
}

func TestNormalizeEmbeddingModelPathExtractsRegistryPathFromRuntimeString(t *testing.T) {
	t.Parallel()

	runtimePath := "MmBertEmbeddingModel(path=models/mmbert-embed-32k-2d-matryoshka, hidden_size=768, layers=22)"
	if got := normalizeEmbeddingModelPath(runtimePath, "mmbert"); got != "models/mmbert-embed-32k-2d-matryoshka" {
		t.Fatalf("expected normalized embedding path, got %q", got)
	}
}

func TestNormalizeEmbeddingModelPathFallsBackToRegistryAlias(t *testing.T) {
	t.Parallel()

	if got := normalizeEmbeddingModelPath("", "gemma"); got != "models/mom-embedding-flash" {
		t.Fatalf("expected alias to resolve to registry local path, got %q", got)
	}
}
