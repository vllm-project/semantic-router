//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func TestStartupEndpointsUseReplicaLocalState(t *testing.T) {
	for _, tc := range []struct {
		name   string
		local  *startupstatus.State
		shared startupstatus.State
		status int
	}{
		{
			name:   "unobserved-replica",
			shared: startupstatus.State{Phase: "ready", Ready: true},
			status: http.StatusServiceUnavailable,
		},
		{
			name:   "starting-replica",
			local:  &startupstatus.State{Phase: "loading", PendingModels: []string{"local-model"}, TotalModels: 2, ReadyModels: 1},
			shared: startupstatus.State{Phase: "ready", Ready: true, TotalModels: 99, ReadyModels: 99},
			status: http.StatusServiceUnavailable,
		},
		{
			name:   "ready-replica",
			local:  &startupstatus.State{Phase: "ready", Ready: true, TotalModels: 2, ReadyModels: 2},
			shared: startupstatus.State{Phase: "loading", PendingModels: []string{"other-model"}},
			status: http.StatusOK,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			configPath := filepath.Join(t.TempDir(), "router.yaml")
			cfg := &config.RouterConfig{StartupStatus: config.StartupStatusConfig{StoreBackend: "file"}}
			local := routerruntime.NewRegistry(cfg)
			other := routerruntime.NewRegistry(cfg)
			localWriter := local.StartupStatusWriter(startupstatus.NewFileWriter(configPath))
			otherWriter := other.StartupStatusWriter(startupstatus.NewFileWriter(configPath))
			if tc.local != nil {
				if err := localWriter.Write(*tc.local); err != nil {
					t.Fatal(err)
				}
			}
			if err := otherWriter.Write(tc.shared); err != nil {
				t.Fatal(err)
			}
			apiServer := &ClassificationAPIServer{config: cfg, configPath: configPath, runtimeRegistry: local}
			mux := http.NewServeMux()
			mux.HandleFunc("/ready", apiServer.handleReady)
			mux.HandleFunc("/startup-status", apiServer.handleStartupStatus)
			server := httptest.NewServer(mux)
			defer server.Close()
			for _, path := range []string{"/ready", "/startup-status"} {
				response, err := server.Client().Get(server.URL + path)
				if err != nil {
					t.Fatal(err)
				}
				body, err := io.ReadAll(response.Body)
				response.Body.Close()
				if err != nil {
					t.Fatal(err)
				}
				t.Logf("%s status=%d body=%s", path, response.StatusCode, body)
				if response.StatusCode != tc.status {
					t.Errorf("%s status=%d, want=%d", path, response.StatusCode, tc.status)
				}
				if tc.local != nil {
					var state startupstatus.State
					if err := json.Unmarshal(body, &state); err != nil {
						t.Fatal(err)
					}
					if state.Phase != tc.local.Phase || state.Ready != tc.local.Ready ||
						state.ReadyModels != tc.local.ReadyModels || state.TotalModels != tc.local.TotalModels {
						t.Errorf("%s returned another replica's state: %+v", path, state)
					}
				}
			}
		})
	}
}

func TestHandleReadyReturns503WhenStatusFileMissing(t *testing.T) {
	tmpDir := t.TempDir()
	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            &config.RouterConfig{},
		configPath:        filepath.Join(tmpDir, "router-config.yaml"),
	}

	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	rr := httptest.NewRecorder()

	apiServer.handleReady(rr, req)

	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 when status file missing, got %d", rr.Code)
	}
}

func TestHandleReadyReturns200WhenStartupReady(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "router-config.yaml")
	if err := startupstatus.NewFileWriter(configPath).Write(startupstatus.State{
		Phase:   "ready",
		Ready:   true,
		Message: "Router startup complete",
	}); err != nil {
		t.Fatalf("failed to write startup status: %v", err)
	}

	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            &config.RouterConfig{},
		configPath:        configPath,
	}

	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	rr := httptest.NewRecorder()

	apiServer.handleReady(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 when startup ready, got %d", rr.Code)
	}
}

func TestHandleReadyUsesSharedStartupStateResolver(t *testing.T) {
	tmpDir := t.TempDir()
	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config: &config.RouterConfig{
			StartupStatus: config.StartupStatusConfig{
				StoreBackend: "redis",
				Redis:        &config.StartupStatusRedisConfig{Address: "127.0.0.1:0"},
			},
		},
		configPath: filepath.Join(tmpDir, "router-config.yaml"),
		startupStateLoader: func() *startupstatus.State {
			return &startupstatus.State{
				Phase:   "ready",
				Ready:   true,
				Message: "Router startup complete from shared status",
			}
		},
	}

	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	rr := httptest.NewRecorder()

	apiServer.handleReady(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 from shared startup resolver, got %d", rr.Code)
	}
}

func TestHandleReadyKeepsBootstrapStatusBackendAcrossRuntimeReload(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "router-config.yaml")
	if err := startupstatus.NewFileWriter(configPath).Write(startupstatus.State{
		Phase: "ready",
		Ready: true,
	}); err != nil {
		t.Fatalf("failed to write startup status: %v", err)
	}

	bootstrapStatus := &config.StartupStatusConfig{StoreBackend: "file"}
	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config: &config.RouterConfig{StartupStatus: config.StartupStatusConfig{
			StoreBackend: "redis",
			Redis:        &config.StartupStatusRedisConfig{Address: "127.0.0.1:0"},
		}},
		configPath:          configPath,
		startupStatusConfig: bootstrapStatus,
	}

	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	rr := httptest.NewRecorder()
	apiServer.handleReady(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected bootstrap file readiness after runtime backend change, got %d", rr.Code)
	}
}
