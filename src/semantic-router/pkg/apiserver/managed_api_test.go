//go:build !windows && cgo

package apiserver

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
)

type managedAPIStub struct {
	readyErr error
	runErr   error
	runCalls int
}

func (managed *managedAPIStub) Register(mux *http.ServeMux) {
	mux.HandleFunc("GET /management/v1/test", func(response http.ResponseWriter, _ *http.Request) {
		response.WriteHeader(http.StatusNoContent)
	})
	mux.HandleFunc("GET /legacy-injected", func(response http.ResponseWriter, _ *http.Request) {
		response.WriteHeader(http.StatusNoContent)
	})
}

func (managed *managedAPIStub) Ready(context.Context) error { return managed.readyErr }

func (managed *managedAPIStub) Run(context.Context) error {
	managed.runCalls++
	return managed.runErr
}

func TestManagedAPICompositionMountsRoutes(t *testing.T) {
	managed := &managedAPIStub{}
	cfg := config.DefaultGlobalConfig()
	cfg.ControlPlane.Mode = config.ControlPlaneModeManaged
	server := &ClassificationAPIServer{managedAPI: managed, config: &cfg}
	request := httptest.NewRequest(http.MethodGet, "/management/v1/test", nil)
	response := httptest.NewRecorder()
	server.setupRoutes().ServeHTTP(response, request)
	if response.Code != http.StatusNoContent {
		t.Fatalf("managed route status = %d", response.Code)
	}
}

func TestManagedListenerExposesOnlyControlPlaneRoutes(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	cfg.ControlPlane.Mode = config.ControlPlaneModeManaged
	server := &ClassificationAPIServer{
		managedAPI: &managedAPIStub{}, config: &cfg,
		startupStateLoader: func() *startupstatus.State {
			return &startupstatus.State{Ready: true, Phase: "ready", Message: "ready"}
		},
	}
	handler := server.setupRoutes()
	for _, path := range []string{"/health", "/ready", "/openapi.json", "/management/v1/test"} {
		request := httptest.NewRequest(http.MethodGet, path, nil)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code == http.StatusNotFound {
			t.Fatalf("managed route %s was not mounted", path)
		}
	}
	for _, path := range []string{
		"/config/router", "/v1/models", "/api/v1/classify/intent", "/startup-status", "/docs", "/legacy-injected",
	} {
		request := httptest.NewRequest(http.MethodGet, path, nil)
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		if response.Code != http.StatusNotFound {
			t.Fatalf("legacy route %s status = %d, want 404", path, response.Code)
		}
	}

	request := httptest.NewRequest(http.MethodGet, "/openapi.json", nil)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	body := response.Body.String()
	if !strings.Contains(body, `"openapi": "3.1.0"`) ||
		!strings.Contains(body, `"/management/v1/providers"`) ||
		strings.Contains(body, `"/config/router"`) {
		t.Fatalf("managed OpenAPI exposed the wrong surface: %s", body)
	}
}

func TestStandaloneListenerKeepsUtilitySurfaceWithoutConfigAuthority(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	server := &ClassificationAPIServer{config: &cfg}
	handler := server.setupRoutes()

	request := httptest.NewRequest(http.MethodGet, "/openapi.json", nil)
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusOK ||
		!strings.Contains(response.Body.String(), `"/api/v1/classify/intent"`) ||
		strings.Contains(response.Body.String(), `"/config/router"`) {
		t.Fatalf("standalone OpenAPI status/body = %d %s", response.Code, response.Body.String())
	}

	request = httptest.NewRequest(http.MethodGet, "/management/v1/test", nil)
	response = httptest.NewRecorder()
	handler.ServeHTTP(response, request)
	if response.Code != http.StatusNotFound {
		t.Fatalf("standalone managed route status = %d, want 404", response.Code)
	}
}

func TestManagedAPIReadinessFailsClosedWithoutExposingDetails(t *testing.T) {
	managed := &managedAPIStub{readyErr: errors.New("database password=secret-value")}
	server := &ClassificationAPIServer{
		managedAPI: managed,
		startupStateLoader: func() *startupstatus.State {
			return &startupstatus.State{Ready: true, Phase: "ready", Message: "Router startup complete"}
		},
	}
	request := httptest.NewRequest(http.MethodGet, "/ready", nil)
	response := httptest.NewRecorder()
	server.handleReady(response, request)
	if response.Code != http.StatusServiceUnavailable {
		t.Fatalf("readiness status = %d, body=%s", response.Code, response.Body.String())
	}
	if body := response.Body.String(); body == "" || strings.Contains(body, "secret-value") {
		t.Fatalf("unsafe readiness body = %s", body)
	}
}

func TestManagedAPIReadinessFailsClosedWithoutListenerTLS(t *testing.T) {
	cfg := config.DefaultGlobalConfig()
	cfg.ControlPlane.Mode = config.ControlPlaneModeManaged
	server := &ClassificationAPIServer{
		managedAPI: &managedAPIStub{}, config: &cfg,
		startupStateLoader: func() *startupstatus.State {
			return &startupstatus.State{Ready: true, Phase: "ready", Message: "ready"}
		},
	}
	request := httptest.NewRequest(http.MethodGet, "/ready", nil)
	response := httptest.NewRecorder()
	server.handleReady(response, request)
	if response.Code != http.StatusServiceUnavailable || !strings.Contains(response.Body.String(), "management_tls") {
		t.Fatalf("missing managed TLS readiness = %d %s", response.Code, response.Body.String())
	}
}
