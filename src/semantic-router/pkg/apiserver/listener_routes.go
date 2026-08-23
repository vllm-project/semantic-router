//go:build !windows && cgo

package apiserver

import (
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

// setupListenerRoutes keeps the standalone utility listener and the managed
// control-plane listener disjoint. Managed mode mounts only the versioned
// Management API and operational probes.
func (s *ClassificationAPIServer) setupListenerRoutes() *http.ServeMux {
	if s.controlPlaneMode() == config.ControlPlaneModeManaged {
		return s.setupManagedListenerRoutes()
	}
	return s.setupStandaloneListenerRoutes()
}

func (s *ClassificationAPIServer) setupStandaloneListenerRoutes() *http.ServeMux {
	mux := http.NewServeMux()
	for _, route := range apiRoutes() {
		mux.HandleFunc(route.pattern(), route.bind(s))
	}
	return mux
}

func (s *ClassificationAPIServer) setupManagedListenerRoutes() *http.ServeMux {
	if s == nil || s.managedAPI == nil {
		panic("managed listener requires a Router-native Management API")
	}
	mux := http.NewServeMux()
	// These operational probes are deliberately independent from the legacy
	// static-token middleware. Domain registrars authenticate every Management
	// operation using Router-issued sessions.
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("GET /ready", s.handleReady)
	mux.HandleFunc("GET /openapi.json", handleManagementOpenAPI)
	managementMux := http.NewServeMux()
	s.managedAPI.Register(managementMux)
	// A registrar cannot widen the managed listener: only the versioned
	// Management prefix is delegated to domain applications.
	mux.Handle("/management/v1/", managementMux)
	return mux
}

func (s *ClassificationAPIServer) controlPlaneMode() string {
	if s == nil {
		return config.ControlPlaneModeStandalone
	}
	cfg := s.currentConfig()
	if cfg == nil || cfg.ControlPlane.Mode == "" {
		return config.ControlPlaneModeStandalone
	}
	return cfg.ControlPlane.Mode
}

func handleManagementOpenAPI(response http.ResponseWriter, _ *http.Request) {
	payload, err := managementapi.GenerateOpenAPIJSON()
	if err != nil {
		http.Error(response, "Management OpenAPI is unavailable.", http.StatusInternalServerError)
		return
	}
	response.Header().Set("Content-Type", "application/json")
	response.Header().Set("Cache-Control", "no-store")
	response.WriteHeader(http.StatusOK)
	_, _ = response.Write(payload)
}
