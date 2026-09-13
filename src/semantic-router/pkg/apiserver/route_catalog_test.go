//go:build !windows && cgo

package apiserver

import (
	"fmt"
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
