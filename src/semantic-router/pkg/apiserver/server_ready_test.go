//go:build !windows && cgo

package apiserver

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/runtimecapabilities"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

type readinessStub struct{ err error }

func (stub *readinessStub) Ready(context.Context) error { return stub.err }

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

func TestHandleReadyFailsClosedUntilDurableRuntimeIsReady(t *testing.T) {
	readiness := &readinessStub{err: errors.New("active routing generation is unavailable")}
	apiServer := &ClassificationAPIServer{
		capabilities:       runtimecapabilities.RuntimeCapabilities{DurableRouting: true},
		runtimeReadiness:   readiness,
		startupStateLoader: func() *startupstatus.State { return &startupstatus.State{Phase: "ready", Ready: true} },
	}

	request := httptest.NewRequest(http.MethodGet, "/ready", nil)
	response := httptest.NewRecorder()
	apiServer.handleReady(response, request)
	if response.Code != http.StatusServiceUnavailable || strings.Contains(response.Body.String(), readiness.err.Error()) {
		t.Fatalf("unready response = %d %s", response.Code, response.Body.String())
	}

	readiness.err = nil
	response = httptest.NewRecorder()
	apiServer.handleReady(response, request)
	if response.Code != http.StatusOK {
		t.Fatalf("ready response = %d %s", response.Code, response.Body.String())
	}
}

func TestHandleReadyFailsClosedWithoutDurableRuntimeReadiness(t *testing.T) {
	apiServer := &ClassificationAPIServer{
		capabilities:       runtimecapabilities.RuntimeCapabilities{DurableRouting: true},
		startupStateLoader: func() *startupstatus.State { return &startupstatus.State{Phase: "ready", Ready: true} },
	}
	response := httptest.NewRecorder()
	apiServer.handleReady(response, httptest.NewRequest(http.MethodGet, "/ready", nil))
	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("missing runtime readiness response = %d %s", response.Code, response.Body.String())
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
