//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

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
	if err := startupstatus.NewWriter(configPath).Write(startupstatus.State{
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
