//go:build !windows && cgo

package apiserver

import (
	"reflect"
	"testing"
)

// TestExportOpenAPISpecMatchesServerGeneration pins the docs-generation export
// to the same output as the runtime /openapi.json handler so docs cannot drift
// from the served spec.
func TestExportOpenAPISpecMatchesServerGeneration(t *testing.T) {
	server := &ClassificationAPIServer{}
	want := server.generateOpenAPISpec()
	got := ExportOpenAPISpec()

	if !reflect.DeepEqual(want, got) {
		t.Fatalf("ExportOpenAPISpec diverged from generateOpenAPISpec: deep-equal failed")
	}
}

func TestExportedRoutesMatchCatalogOrder(t *testing.T) {
	want := apiEndpointMetadata()
	got := ExportedRoutes()

	if len(want) != len(got) {
		t.Fatalf("expected %d exported routes, got %d", len(want), len(got))
	}
	for i, route := range want {
		if got[i].Path != route.Path || got[i].Method != route.Method || got[i].Description != route.Description {
			t.Fatalf("route %d mismatch: expected %#v, got %#v", i, route, got[i])
		}
	}
}
