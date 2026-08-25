//go:build !windows && cgo

package apiserver

import (
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/managementapi"
)

// setupListenerRoutes keeps the file-authoritative utility surface, durable
// operational probes, and the explicitly enabled Management API disjoint.
func (s *ClassificationAPIServer) setupListenerRoutes() *http.ServeMux {
	if s != nil && s.capabilities.ManagementAPI {
		return s.setupManagementListenerRoutes()
	}
	if s != nil && s.capabilities.DurableRouting {
		return s.setupOperationalListenerRoutes()
	}
	return s.setupFileListenerRoutes()
}

func (s *ClassificationAPIServer) setupOperationalListenerRoutes() *http.ServeMux {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("GET /ready", s.handleReady)
	return mux
}

func (s *ClassificationAPIServer) setupFileListenerRoutes() *http.ServeMux {
	mux := http.NewServeMux()
	for _, route := range apiRoutes() {
		mux.HandleFunc(route.pattern(), route.bind(s))
	}
	return mux
}

func (s *ClassificationAPIServer) setupManagementListenerRoutes() *http.ServeMux {
	if s == nil || s.managementAPI == nil {
		panic("Management listener requires a Router-native Management API")
	}
	mux := http.NewServeMux()
	// These operational probes are deliberately independent from the legacy
	// static-token middleware. Domain registrars authenticate every Management
	// operation using Router-issued sessions.
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("GET /ready", s.handleReady)
	mux.HandleFunc("GET /openapi.json", handleManagementOpenAPI)
	managementMux := http.NewServeMux()
	s.managementAPI.Register(managementMux)
	// A registrar cannot widen the Management listener: only the versioned
	// Management prefix is delegated to domain applications.
	mux.Handle("/management/v1/", managementMux)
	return mux
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
