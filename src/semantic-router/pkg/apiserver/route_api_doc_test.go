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
	response := requestAPIOverview(t, "/api/v1")

	if response.Service == "" {
		t.Error("expected non-empty service name")
	}
	if response.Version != "v1" {
		t.Errorf("expected version 'v1', got %q", response.Version)
	}
	if len(response.Capabilities) == 0 {
		t.Error("expected at least one capability")
	}
	if len(response.Endpoints) != 0 {
		t.Fatalf("compact discovery unexpectedly returned %d endpoints", len(response.Endpoints))
	}
	if len(response.Links) == 0 {
		t.Error("expected at least one link")
	}

	assertCapabilities(t, response, []string{"system", "config", "routing", "inventory", "observability", "storage", "diagnostics"})
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
		[]byte("openapi.json"),
		[]byte("SwaggerUIBundle"),
	} {
		if !bytes.Contains(html, snippet) {
			t.Errorf("expected HTML to contain %q", snippet)
		}
	}
}

func TestAPIOverviewIncludesNewEndpoints(t *testing.T) {
	response := requestAPIOverview(t, "/api/v1?view=operations")

	assertOverviewPaths(t, response, []string{"/openapi.json", "/docs"})
	if response.Links["openapi_spec"] != "/openapi.json" {
		t.Error("expected 'openapi_spec' link to '/openapi.json'")
	}
	if response.Links["swagger_ui"] != "/docs" {
		t.Error("expected 'swagger_ui' link to '/docs'")
	}
	if response.Links["openapi_operation"] != "/openapi.json?path={path}&method={method}" {
		t.Error("expected progressive OpenAPI operation link")
	}
	if response.Links["config_schema"] != "/api/v1/config/schema" {
		t.Error("expected direct Router config schema link")
	}
}

func TestAPIOverviewIncludesRoutePolicyMetadata(t *testing.T) {
	response := requestAPIOverview(t, "/api/v1?capability=config")
	for _, endpoint := range response.Endpoints {
		if endpoint.Path != "/api/v1/config" || endpoint.Method != http.MethodPatch {
			continue
		}
		if endpoint.Permission != PermConfigWrite || endpoint.Sensitivity != SensitivityMutation {
			t.Fatalf("config patch policy metadata = %+v", endpoint)
		}
		if endpoint.Capability != "config" || endpoint.Plane != APIPlaneManagement || !containsAudience(endpoint.Audiences, APIAudienceAgent) {
			t.Fatalf("config patch contract metadata = %+v", endpoint.EndpointContract)
		}
		return
	}
	t.Fatal("config patch endpoint is missing")
}

func TestAPIOverviewFiltersOneCapability(t *testing.T) {
	response := requestAPIOverview(t, "/api/v1?capability=routing")
	if len(response.Endpoints) != 1 {
		t.Fatalf("routing discovery endpoints = %d, want 1", len(response.Endpoints))
	}
	if response.Endpoints[0].Path != apiRoutingPreviewPath {
		t.Fatalf("routing endpoint = %q", response.Endpoints[0].Path)
	}
	if len(response.Capabilities) != 1 || response.Capabilities[0].Name != "routing" {
		t.Fatalf("routing capabilities = %+v", response.Capabilities)
	}
}

func TestAPIOverviewRejectsUnknownFilter(t *testing.T) {
	apiServer := newDocumentationTestServer()
	req := httptest.NewRequest(http.MethodGet, "/api/v1?capability=unknown", nil)
	rr := httptest.NewRecorder()
	apiServer.handleAPIOverview(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("unknown capability status = %d, want 400", rr.Code)
	}
}

func TestAPIOverviewFiltersAgentPrimaryOperations(t *testing.T) {
	response := requestAPIOverview(t, "/api/v1?view=operations&audience=agent&visibility=primary")
	if len(response.Endpoints) == 0 {
		t.Fatal("expected primary agent operations")
	}
	for _, endpoint := range response.Endpoints {
		if endpoint.Visibility != APIVisibilityPrimary || !containsAudience(endpoint.Audiences, APIAudienceAgent) {
			t.Fatalf("unexpected endpoint in agent-primary view: %+v", endpoint)
		}
	}
}

func newDocumentationTestServer() *ClassificationAPIServer {
	return &ClassificationAPIServer{
		classificationSvc: services.NewPlaceholderClassificationService(),
		config:            &config.RouterConfig{},
	}
}

func requestAPIOverview(t *testing.T, target string) APIOverviewResponse {
	t.Helper()

	apiServer := newDocumentationTestServer()
	req := httptest.NewRequest(http.MethodGet, target, nil)
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

func assertCapabilities(t *testing.T, response APIOverviewResponse, expected []string) {
	t.Helper()

	found := make(map[string]bool, len(expected))
	for _, capability := range response.Capabilities {
		found[capability.Name] = true
	}
	for _, capability := range expected {
		if !found[capability] {
			t.Errorf("expected to find capability %q in response", capability)
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
		"/api/v1/diagnostics/classify/intent",
		"/api/v1/diagnostics/classify/pii",
		"/api/v1/diagnostics/classify/security",
		"/api/v1/diagnostics/classify/batch",
		"/api/v1/routing/preview",
		"/api/v1/diagnostics/nli",
		"/api/v1/diagnostics/embeddings",
		"/api/v1/diagnostics/similarity/batch",
		"/health",
		"/ready",
		"/startup-status",
		"/api/v1/config",
		"/api/v1/config/schema",
		"/api/v1/config/validate",
		"/api/v1/config/plan",
		"/api/v1/config/rollback",
		"/api/v1/config/versions",
		"/api/v1/config/recipes",
		"/api/v1/config/recipes/validate",
		"/api/v1/config/recipes/{name}",
		"/api/v1/config/hash",
		"/api/v1/storage/memories",
		"/api/v1/storage/vector-stores",
		"/api/v1/storage/files",
	}
}
