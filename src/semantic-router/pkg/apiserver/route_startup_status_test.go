//go:build !windows && cgo

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
