//go:build !windows

package apiserver

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

func TestHandleStartupStatusReturns503WhenNoStatus(t *testing.T) {
	tmpDir := t.TempDir()
	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            &config.RouterConfig{},
		configPath:        filepath.Join(tmpDir, "router-config.yaml"),
	}

	req := httptest.NewRequest(http.MethodGet, "/startup-status", nil)
	rr := httptest.NewRecorder()

	apiServer.handleStartupStatus(rr, req)

	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 when no status available, got %d", rr.Code)
	}
}

func TestHandleStartupStatusReturns200WhenReady(t *testing.T) {
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

	req := httptest.NewRequest(http.MethodGet, "/startup-status", nil)
	rr := httptest.NewRecorder()

	apiServer.handleStartupStatus(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 when startup ready, got %d", rr.Code)
	}

	var state startupstatus.State
	if err := json.Unmarshal(rr.Body.Bytes(), &state); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}
	if !state.Ready {
		t.Fatalf("expected Ready=true, got false")
	}
	if state.Phase != "ready" {
		t.Fatalf("expected Phase=%q, got %q", "ready", state.Phase)
	}
}

func TestHandleStartupStatusReturnsEmbeddingProviderStatus(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "router-config.yaml")
	apiKeyEnvSet := true
	healthy := true
	if err := startupstatus.NewFileWriter(configPath).Write(startupstatus.State{
		Phase:   "ready",
		Ready:   true,
		Message: "Router startup complete",
		EmbeddingProvider: &startupstatus.EmbeddingProviderStatus{
			Mode:          "remote",
			Backend:       config.EmbeddingBackendOpenAICompatible,
			Model:         "text-embedding-3-small",
			Dimension:     1536,
			APIKeyEnv:     "OPENAI_API_KEY",
			APIKeyEnvSet:  &apiKeyEnvSet,
			Healthy:       &healthy,
			LastCheckedAt: "2026-07-08T00:00:00Z",
		},
	}); err != nil {
		t.Fatalf("failed to write startup status: %v", err)
	}

	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            &config.RouterConfig{},
		configPath:        configPath,
	}

	req := httptest.NewRequest(http.MethodGet, "/startup-status", nil)
	rr := httptest.NewRecorder()

	apiServer.handleStartupStatus(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 when startup ready, got %d", rr.Code)
	}

	var state startupstatus.State
	if err := json.Unmarshal(rr.Body.Bytes(), &state); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}
	if state.EmbeddingProvider == nil {
		t.Fatal("expected embedding provider status")
	}
	if state.EmbeddingProvider.APIKeyEnv != "OPENAI_API_KEY" {
		t.Fatalf("api key env = %q", state.EmbeddingProvider.APIKeyEnv)
	}
	if state.EmbeddingProvider.APIKeyEnvSet == nil || !*state.EmbeddingProvider.APIKeyEnvSet {
		t.Fatalf("api key env set = %v, want true", state.EmbeddingProvider.APIKeyEnvSet)
	}

	var raw map[string]interface{}
	if err := json.Unmarshal(rr.Body.Bytes(), &raw); err != nil {
		t.Fatalf("failed to unmarshal raw response: %v", err)
	}
	embeddingProvider, ok := raw["embedding_provider"].(map[string]interface{})
	if !ok {
		t.Fatalf("embedding_provider missing from raw response: %+v", raw)
	}
	if _, ok := embeddingProvider["api_key"]; ok {
		t.Fatalf("startup status leaked api_key field: %+v", embeddingProvider)
	}
}

func TestHandleStartupStatusReturns503WhenDownloading(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "router-config.yaml")
	if err := startupstatus.NewFileWriter(configPath).Write(startupstatus.State{
		Phase:            "downloading_models",
		Ready:            false,
		Message:          "Downloading models...",
		DownloadingModel: "models/test",
		TotalModels:      3,
		ReadyModels:      1,
	}); err != nil {
		t.Fatalf("failed to write startup status: %v", err)
	}

	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            &config.RouterConfig{},
		configPath:        configPath,
	}

	req := httptest.NewRequest(http.MethodGet, "/startup-status", nil)
	rr := httptest.NewRecorder()

	apiServer.handleStartupStatus(rr, req)

	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 when downloading, got %d", rr.Code)
	}

	var state startupstatus.State
	if err := json.Unmarshal(rr.Body.Bytes(), &state); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}
	if state.Ready {
		t.Fatalf("expected Ready=false, got true")
	}
	if state.DownloadingModel != "models/test" {
		t.Fatalf("expected DownloadingModel=%q, got %q", "models/test", state.DownloadingModel)
	}
}

func TestReadinessNamesTheModelDeploymentsStartupWaitsFor(t *testing.T) {
	apiServer := &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            &config.RouterConfig{},
		startupStateLoader: func() *startupstatus.State {
			return &startupstatus.State{
				Phase: startupstatus.PhaseLoadingModelDeployments, Message: "Waiting for Router-managed model deployments",
				PendingModels: []string{"decider"}, TotalModels: 1,
				ModelDeployments: []startupstatus.ModelDeploymentStatus{
					{Name: "decider", Artifact: "vllm-sr/Decision-2.0-Kai-0.6B", Process: "cpu", State: "loading"},
				},
			}
		},
	}

	rr := httptest.NewRecorder()
	apiServer.handleReady(rr, httptest.NewRequest(http.MethodGet, "/ready", nil))
	var ready map[string]interface{}
	if err := json.Unmarshal(rr.Body.Bytes(), &ready); err != nil {
		t.Fatal(err)
	}
	if rr.Code != http.StatusServiceUnavailable || ready["ready"] != false || ready["phase"] != startupstatus.PhaseLoadingModelDeployments ||
		ready["total_models"] != float64(1) || ready["ready_models"] != float64(0) {
		t.Fatalf("/ready while a deployment loads: %d %s", rr.Code, rr.Body.String())
	}
	if pending, _ := ready["pending_models"].([]interface{}); len(pending) != 1 || pending[0] != "decider" {
		t.Fatalf("/ready names the deployment still loading: %s", rr.Body.String())
	}

	rr = httptest.NewRecorder()
	apiServer.handleStartupStatus(rr, httptest.NewRequest(http.MethodGet, "/startup-status", nil))
	var status startupstatus.State
	if err := json.Unmarshal(rr.Body.Bytes(), &status); err != nil {
		t.Fatal(err)
	}
	if rr.Code != http.StatusServiceUnavailable || len(status.ModelDeployments) != 1 ||
		status.ModelDeployments[0] != (startupstatus.ModelDeploymentStatus{Name: "decider", Artifact: "vllm-sr/Decision-2.0-Kai-0.6B", Process: "cpu", State: "loading"}) {
		t.Fatalf("/startup-status lists the deployment with its state: %d %s", rr.Code, rr.Body.String())
	}
}
