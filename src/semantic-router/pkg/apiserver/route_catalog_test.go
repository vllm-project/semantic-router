//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestAPIRouteCatalogHasUniqueDocumentedPatterns(t *testing.T) {
	routes := apiRoutes()
	metadata := apiEndpointMetadata()
	if len(routes) != len(metadata) {
		t.Fatalf("expected metadata count %d to match route count %d", len(metadata), len(routes))
	}

	seen := make(map[string]struct{}, len(routes))
	for _, route := range routes {
		if route.Handler == nil {
			t.Fatalf("route %s has no handler", route.pattern())
		}
		if route.Capability == "" || !knownCapability(route.Capability) {
			t.Fatalf("route %s has unknown capability %q", route.pattern(), route.Capability)
		}
		if !oneOfPlane(route.Plane) || !oneOfVisibility(route.Visibility) || len(route.Audiences) == 0 {
			t.Fatalf("route %s has incomplete semantic contract: %+v", route.pattern(), route.EndpointContract)
		}
		for _, audience := range route.Audiences {
			if !oneOfAudience(audience) {
				t.Fatalf("route %s has unknown audience %q", route.pattern(), audience)
			}
		}
		if route.Stability != APIStabilityStable && route.Stability != APIStabilityExperimental {
			t.Fatalf("route %s has unknown stability %q", route.pattern(), route.Stability)
		}
		if route.Path != "/v1/models" && !strings.HasPrefix(route.Path, apiRootPath) && route.Path != "/health" && route.Path != "/ready" && route.Path != "/startup-status" && route.Path != "/openapi.json" && route.Path != "/docs" {
			t.Fatalf("route %s is outside the canonical API namespace", route.pattern())
		}
		for _, retired := range []string{
			"/api/v1/response-cache",
			"/api/v1/context-compression",
			"/api/v1/classify",
			"/api/v1/eval",
			"/api/v1/nli",
			"/api/v1/embeddings",
			"/api/v1/similarity",
			"/config",
			"/info",
			"/metrics",
			"/v1/router",
			"/v1/router_replay",
			"/v1/memory",
			"/v1/files",
			"/v1/vector_stores",
		} {
			if route.Path == retired || strings.HasPrefix(route.Path, retired+"/") {
				t.Fatalf("retired route remains registered: %s", route.pattern())
			}
		}

		key := fmt.Sprintf("%s %s", route.Method, route.Path)
		if _, exists := seen[key]; exists {
			t.Fatalf("duplicate route pattern %q", key)
		}
		seen[key] = struct{}{}
	}
}

// Retired paths must not redirect a mutation or reach a compatibility handler.
func TestManagementResourceBoundariesRejectRetiredOperations(t *testing.T) {
	mux := (&ClassificationAPIServer{}).setupRoutes()
	for _, path := range []string{
		"/config/router", "/api/v1/response-cache/stats", "/api/v1/response-cache/flush",
		"/api/v1/response-cache/audit", "/api/v1/context-compression/preview",
		"/api/v1/context-compression/stats", "/api/v1/context-compression/recovery/invalidate",
	} {
		for _, method := range []string{http.MethodGet, http.MethodPost} {
			response := httptest.NewRecorder()
			mux.ServeHTTP(response, httptest.NewRequest(method, path, nil))
			if response.Code != http.StatusNotFound {
				t.Fatalf("retired %s %s = %d, want 404", method, path, response.Code)
			}
		}
	}
	for _, capability := range capabilityRegistry {
		if capability.Name == "response-cache" || capability.Name == "context-compression" {
			t.Fatalf("implementation feature leaked into top-level resource taxonomy: %s", capability.Name)
		}
	}
}
