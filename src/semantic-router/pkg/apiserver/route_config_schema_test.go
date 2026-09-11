//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configschema"
)

func TestConfigSchemaRouteDefaultsToCompactIndex(t *testing.T) {
	server := &ClassificationAPIServer{}
	response := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, configschema.SchemaEndpoint, nil)
	server.handleConfigSchema(response, request)

	if response.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", response.Code, response.Body.String())
	}
	if got := response.Header().Get("Content-Type"); got != "application/json" {
		t.Fatalf("content type=%q", got)
	}
	var document map[string]any
	if err := json.Unmarshal(response.Body.Bytes(), &document); err != nil {
		t.Fatalf("invalid schema JSON: %v", err)
	}
	if document["default_view"] != configschema.ViewIndex {
		t.Fatalf("default view=%v", document["default_view"])
	}
	etag := response.Header().Get("ETag")
	if etag == "" {
		t.Fatal("schema index response did not expose its content identity")
	}

	notModified := httptest.NewRecorder()
	cachedRequest := httptest.NewRequest(http.MethodGet, configschema.SchemaEndpoint, nil)
	cachedRequest.Header.Set("If-None-Match", etag)
	server.handleConfigSchema(notModified, cachedRequest)
	if notModified.Code != http.StatusNotModified || notModified.Body.Len() != 0 {
		t.Fatalf("conditional response status=%d body=%q", notModified.Code, notModified.Body.String())
	}
}

func TestConfigSchemaRouteSupportsProgressiveViews(t *testing.T) {
	server := &ClassificationAPIServer{}

	full := httptest.NewRecorder()
	server.handleConfigSchema(full, httptest.NewRequest(http.MethodGet, "/api/v1/config/schema?view=full", nil))
	if full.Code != http.StatusOK || full.Header().Get("Content-Type") != "application/schema+json" {
		t.Fatalf("full status=%d content-type=%q", full.Code, full.Header().Get("Content-Type"))
	}
	if full.Header().Get("ETag") != configschema.ETag() {
		t.Fatal("full schema response did not expose its generated content identity")
	}
	var fullDocument map[string]any
	if err := json.Unmarshal(full.Body.Bytes(), &fullDocument); err != nil {
		t.Fatalf("decode full schema: %v", err)
	}
	if fullDocument["$id"] != configschema.SchemaID {
		t.Fatalf("schema id=%v", fullDocument["$id"])
	}

	index := httptest.NewRecorder()
	server.handleConfigSchema(index, httptest.NewRequest(http.MethodGet, "/api/v1/config/schema?view=index", nil))
	if index.Code != http.StatusOK {
		t.Fatalf("index status=%d body=%s", index.Code, index.Body.String())
	}
	if index.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("index content type=%q", index.Header().Get("Content-Type"))
	}
	var directory map[string]any
	if err := json.Unmarshal(index.Body.Bytes(), &directory); err != nil {
		t.Fatalf("decode index: %v", err)
	}
	if directory["default_view"] != configschema.ViewIndex {
		t.Fatalf("default view=%v", directory["default_view"])
	}

	section := httptest.NewRecorder()
	server.handleConfigSchema(section, httptest.NewRequest(http.MethodGet, "/api/v1/config/schema?view=section&path=global.router.learning", nil))
	if section.Code != http.StatusOK {
		t.Fatalf("section status=%d body=%s", section.Code, section.Body.String())
	}
	if section.Body.Len() >= len(configschema.Document()) {
		t.Fatalf("section view should be smaller than full schema: section=%d full=%d", section.Body.Len(), len(configschema.Document()))
	}

	surface := httptest.NewRecorder()
	server.handleConfigSchema(surface, httptest.NewRequest(http.MethodGet, "/api/v1/config/schema?view=surface&kind=algorithm&name=static", nil))
	if surface.Code != http.StatusOK {
		t.Fatalf("surface status=%d body=%s", surface.Code, surface.Body.String())
	}

	invalid := httptest.NewRecorder()
	server.handleConfigSchema(invalid, httptest.NewRequest(http.MethodGet, "/api/v1/config/schema?view=section&path=not.real", nil))
	if invalid.Code != http.StatusBadRequest {
		t.Fatalf("invalid section status=%d, want %d", invalid.Code, http.StatusBadRequest)
	}
}
