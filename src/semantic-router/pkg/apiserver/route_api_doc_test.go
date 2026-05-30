//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestAPIOverviewEndpoint(t *testing.T) {
	response := requestAPIOverview(t)

	if response.Service == "" {
		t.Error("expected non-empty service name")
	}
	if response.Version != "v1" {
		t.Errorf("expected version 'v1', got %q", response.Version)
	}
	if len(response.Endpoints) == 0 {
		t.Error("expected at least one endpoint")
	}
	if len(response.Links) == 0 {
		t.Error("expected at least one link")
	}

	assertTaskTypes(t, response.TaskTypes, []string{"intent", "pii", "security", "all"})
	assertOverviewPaths(t, response, documentedAPIOverviewPaths())
	assertOverviewPathsAbsent(t, response, []string{
		"/config/classification",
		"/config/system-prompts",
		"/config/deploy",
	})
}

func TestSwaggerUIEndpoint(t *testing.T) {
	apiServer := newDocumentationTestServer()
	req := httptest.NewRequest(http.MethodGet, "/docs", nil)
	rr := httptest.NewRecorder()

	apiServer.handleSwaggerUI(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", rr.Code)
	}
	if contentType := rr.Header().Get("Content-Type"); contentType != "text/html; charset=utf-8" {
		t.Errorf("expected Content-Type 'text/html; charset=utf-8', got %q", contentType)
	}

	html := rr.Body.Bytes()
	for _, snippet := range [][]byte{
		[]byte("swagger-ui"),
		[]byte("/openapi.json"),
		[]byte("SwaggerUIBundle"),
	} {
		if !bytes.Contains(html, snippet) {
			t.Errorf("expected HTML to contain %q", snippet)
		}
	}
}

func TestAPIOverviewIncludesNewEndpoints(t *testing.T) {
	response := requestAPIOverview(t)

	assertOverviewPaths(t, response, []string{"/openapi.json", "/docs"})
	if response.Links["openapi_spec"] != "/openapi.json" {
		t.Error("expected 'openapi_spec' link to '/openapi.json'")
	}
	if response.Links["swagger_ui"] != "/docs" {
		t.Error("expected 'swagger_ui' link to '/docs'")
	}
}

func newDocumentationTestServer() *ClassificationAPIServer {
	return &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            &config.RouterConfig{},
	}
}

func requestAPIOverview(t *testing.T) APIOverviewResponse {
	t.Helper()

	apiServer := newDocumentationTestServer()
	req := httptest.NewRequest(http.MethodGet, "/api/v1", nil)
	rr := httptest.NewRecorder()

	apiServer.handleAPIOverview(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", rr.Code)
	}

	var response APIOverviewResponse
	if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}
	return response
}

func assertTaskTypes(t *testing.T, taskTypes []TaskTypeInfo, expected []string) {
	t.Helper()

	found := make(map[string]bool, len(expected))
	for _, taskType := range taskTypes {
		found[taskType.Name] = true
	}
	for _, taskType := range expected {
		if !found[taskType] {
			t.Errorf("expected to find task_type %q in response", taskType)
		}
	}
}

func assertOverviewPaths(t *testing.T, response APIOverviewResponse, expected []string) {
	t.Helper()

	endpointPaths := make(map[string]bool, len(response.Endpoints))
	for _, endpoint := range response.Endpoints {
		endpointPaths[endpoint.Path] = true
	}
	for _, path := range expected {
		if !endpointPaths[path] {
			t.Errorf("expected to find endpoint %q in response", path)
		}
	}
}

func assertOverviewPathsAbsent(t *testing.T, response APIOverviewResponse, absent []string) {
	t.Helper()

	endpointPaths := make(map[string]bool, len(response.Endpoints))
	for _, endpoint := range response.Endpoints {
		endpointPaths[endpoint.Path] = true
	}
	for _, path := range absent {
		if endpointPaths[path] {
			t.Errorf("expected endpoint %q to be absent", path)
		}
	}
}

func documentedAPIOverviewPaths() []string {
	return []string{
		"/api/v1/classify/intent",
		"/api/v1/classify/pii",
		"/api/v1/classify/security",
		"/api/v1/classify/batch",
		"/api/v1/eval",
		"/api/v1/nli",
		"/api/v1/embeddings",
		"/api/v1/similarity/batch",
		"/health",
		"/ready",
		"/startup-status",
		"/config/router",
		"/config/router/rollback",
		"/config/router/versions",
		"/config/hash",
		"/v1/memory",
		"/v1/vector_stores",
		"/v1/files",
	}
}
