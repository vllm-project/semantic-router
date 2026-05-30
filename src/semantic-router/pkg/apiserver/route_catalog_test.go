//go:build !windows && cgo

package apiserver

import (
	"fmt"
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

		key := fmt.Sprintf("%s %s", route.Method, route.Path)
		if _, exists := seen[key]; exists {
			t.Fatalf("duplicate route pattern %q", key)
		}
		seen[key] = struct{}{}
	}
}
