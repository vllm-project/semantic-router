//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOpenAPISpecUsesRouteBodyMetadata(t *testing.T) {
	apiServer := &ClassificationAPIServer{}
	spec := apiServer.generateOpenAPISpec()

	upload := spec.Paths["/v1/files"].Post
	if upload == nil || upload.RequestBody == nil {
		t.Fatalf("expected /v1/files POST request body metadata")
	}
	if _, ok := upload.RequestBody.Content[string(requestBodyMultipart)]; !ok {
		t.Fatalf("expected /v1/files POST to use multipart request body metadata")
	}
	if _, ok := upload.RequestBody.Content[string(requestBodyJSON)]; ok {
		t.Fatalf("did not expect /v1/files POST to advertise JSON request body metadata")
	}
	if _, ok := upload.Responses["413"]; !ok {
		t.Fatalf("expected /v1/files POST to document request-size limit response")
	}

	search := spec.Paths["/v1/vector_stores/{id}/search"].Post
	if search == nil || search.RequestBody == nil {
		t.Fatalf("expected vector-store search request body metadata")
	}
	if !strings.Contains(search.RequestBody.Description, fmt.Sprintf("%d", maxVectorStoreJSONBodySize)) {
		t.Fatalf("expected vector-store search to document %d byte limit, got %q", maxVectorStoreJSONBodySize, search.RequestBody.Description)
	}
}

func TestOpenAPISpecDerivesPathParametersFromRoutes(t *testing.T) {
	apiServer := &ClassificationAPIServer{}
	spec := apiServer.generateOpenAPISpec()

	detach := spec.Paths["/v1/vector_stores/{id}/files/{file_id}"].Delete
	if detach == nil {
		t.Fatalf("expected vector-store file detach operation")
	}

	requireOpenAPIPathParameter(t, detach.Parameters, "id")
	requireOpenAPIPathParameter(t, detach.Parameters, "file_id")
	if strings.ContainsAny(detach.OperationID, "/{}-.") {
		t.Fatalf("expected sanitized operation ID, got %q", detach.OperationID)
	}

	listFiles := spec.Paths["/v1/files"].Get
	if listFiles == nil {
		t.Fatalf("expected file list operation")
	}
	if len(listFiles.Parameters) != 0 {
		t.Fatalf("expected no path parameters for /v1/files, got %+v", listFiles.Parameters)
	}
}

func TestOpenAPISpecEndpoint(t *testing.T) {
	apiServer := newDocumentationTestServer()
	req := httptest.NewRequest(http.MethodGet, "/openapi.json", nil)
	rr := httptest.NewRecorder()

	apiServer.handleOpenAPISpec(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", rr.Code)
	}
	if contentType := rr.Header().Get("Content-Type"); contentType != "application/json" {
		t.Errorf("expected Content-Type 'application/json', got %q", contentType)
	}

	var spec OpenAPISpec
	if err := json.Unmarshal(rr.Body.Bytes(), &spec); err != nil {
		t.Fatalf("failed to unmarshal OpenAPI spec: %v", err)
	}

	assertOpenAPISpecBasics(t, spec)
	assertOpenAPIPaths(t, spec, documentedOpenAPIPaths())
	assertRouterConfigOpenAPIPath(t, spec)
	assertOpenAPIPathsAbsent(t, spec, []string{
		"/config/classification",
		"/config/system-prompts",
	})
}

func assertOpenAPISpecBasics(t *testing.T, spec OpenAPISpec) {
	t.Helper()

	if spec.OpenAPI != "3.0.0" {
		t.Errorf("expected OpenAPI version '3.0.0', got %q", spec.OpenAPI)
	}
	if spec.Info.Title == "" {
		t.Error("expected non-empty title")
	}
	if spec.Info.Version != "v1" {
		t.Errorf("expected version 'v1', got %q", spec.Info.Version)
	}
	if len(spec.Paths) == 0 {
		t.Error("expected at least one path in OpenAPI spec")
	}
}

func assertOpenAPIPaths(t *testing.T, spec OpenAPISpec, expected []string) {
	t.Helper()

	for _, path := range expected {
		if _, exists := spec.Paths[path]; !exists {
			t.Errorf("expected path %q to be in OpenAPI spec", path)
		}
	}
}

func assertOpenAPIPathsAbsent(t *testing.T, spec OpenAPISpec, absent []string) {
	t.Helper()

	for _, path := range absent {
		if _, exists := spec.Paths[path]; exists {
			t.Errorf("expected path %q to be absent from OpenAPI spec", path)
		}
	}
}

func assertRouterConfigOpenAPIPath(t *testing.T, spec OpenAPISpec) {
	t.Helper()

	routerPath, exists := spec.Paths["/config/router"]
	if !exists {
		t.Fatalf("expected /config/router to be documented in OpenAPI spec")
	}
	if routerPath.Patch == nil || routerPath.Put == nil || routerPath.Get == nil {
		t.Fatalf("expected /config/router to document GET, PATCH, and PUT, got %+v", routerPath)
	}
	if _, ok := routerPath.Patch.Responses["413"]; !ok {
		t.Fatalf("expected /config/router PATCH to document 413 request body limit response")
	}
	if routerPath.Patch.RequestBody == nil || routerPath.Patch.RequestBody.Description == "" {
		t.Fatalf("expected /config/router PATCH to document request body constraints")
	}
}

func requireOpenAPIPathParameter(t *testing.T, parameters []OpenAPIParameter, name string) {
	t.Helper()

	for _, parameter := range parameters {
		if parameter.Name != name {
			continue
		}
		if parameter.In != "path" {
			t.Fatalf("expected %q parameter location path, got %q", name, parameter.In)
		}
		if !parameter.Required {
			t.Fatalf("expected %q path parameter to be required", name)
		}
		if parameter.Schema.Type != "string" {
			t.Fatalf("expected %q path parameter schema string, got %+v", name, parameter.Schema)
		}
		return
	}

	t.Fatalf("expected %q path parameter in %+v", name, parameters)
}

func documentedOpenAPIPaths() []string {
	return []string{
		"/health",
		"/ready",
		"/startup-status",
		"/api/v1",
		"/api/v1/classify/batch",
		"/api/v1/eval",
		"/api/v1/nli",
		"/api/v1/embeddings",
		"/api/v1/similarity/batch",
		"/openapi.json",
		"/docs",
		"/config/router",
		"/config/router/rollback",
		"/config/router/versions",
		"/config/hash",
		"/v1/memory",
		"/v1/vector_stores",
		"/v1/files",
	}
}
