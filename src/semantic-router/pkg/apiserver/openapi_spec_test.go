//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configschema"
)

func TestKnowledgeBaseOpenAPIDocumentsPendingPublication(t *testing.T) {
	spec := (&ClassificationAPIServer{}).generateOpenAPISpec()
	collection := spec.Paths[apiStorageKnowledgeBasesPath]
	item := spec.Paths[apiStorageKnowledgeBasesPath+"/{name}"]
	for _, operation := range []*OpenAPIOperation{collection.Post, item.Put, item.Delete} {
		if operation == nil {
			t.Fatal("missing knowledge-base mutation operation")
		}
		pending, ok := operation.Responses["202"]
		if !ok || pending.Content["application/json"].Schema.Properties["generated_runtime_hash"].Type != "string" {
			t.Fatalf("pending response missing exact candidate hash: %+v", operation.Responses)
		}
		if !strings.Contains(operation.Responses["409"].Description, "CONFIG_ACTIVATION_PENDING") {
			t.Fatal("pending mutation conflict was not documented")
		}
	}
	if _, ok := collection.Post.Responses["201"]; !ok {
		t.Fatal("create success status was not documented")
	}
	if _, ok := collection.Get.Responses["202"]; ok {
		t.Fatal("read operation incorrectly documents candidate persistence")
	}
}

func TestOpenAPISpecUsesRouteBodyMetadata(t *testing.T) {
	apiServer := &ClassificationAPIServer{}
	spec := apiServer.generateOpenAPISpec()

	upload := spec.Paths["/api/v1/storage/files"].Post
	if upload == nil || upload.RequestBody == nil {
		t.Fatalf("expected /api/v1/storage/files POST request body metadata")
	}
	if _, ok := upload.RequestBody.Content[string(requestBodyMultipart)]; !ok {
		t.Fatalf("expected /api/v1/storage/files POST to use multipart request body metadata")
	}
	if _, ok := upload.RequestBody.Content[string(requestBodyJSON)]; ok {
		t.Fatalf("did not expect /api/v1/storage/files POST to advertise JSON request body metadata")
	}
	if _, ok := upload.Responses["413"]; !ok {
		t.Fatalf("expected /api/v1/storage/files POST to document request-size limit response")
	}

	search := spec.Paths["/api/v1/storage/vector-stores/{id}/search"].Post
	if search == nil || search.RequestBody == nil {
		t.Fatalf("expected vector-store search request body metadata")
	}
	if !strings.Contains(search.RequestBody.Description, fmt.Sprintf("%d", maxVectorStoreJSONBodySize)) {
		t.Fatalf("expected vector-store search to document %d byte limit, got %q", maxVectorStoreJSONBodySize, search.RequestBody.Description)
	}

	intent := spec.Paths["/api/v1/diagnostics/classify/intent"].Post
	intentSchema := intent.RequestBody.Content[string(requestBodyJSON)].Schema
	if intentSchema == nil || intentSchema.Properties["text"].Type != "string" {
		t.Fatalf("expected intent body schema to come from services.IntentRequest, got %+v", intentSchema)
	}
	if slices.Contains(intentSchema.Required, "text") {
		t.Fatalf("text is optional when messages supply the signal input, got required=%v", intentSchema.Required)
	}

	configPatch := spec.Paths["/api/v1/config"].Patch
	configSchema := configPatch.RequestBody.Content[string(requestBodyJSON)].Schema
	if configSchema == nil || configSchema.Properties["yaml"].Type != "string" {
		t.Fatalf("expected config body schema to come from RouterConfigUpdateRequest, got %+v", configSchema)
	}
	if _, exists := configSchema.Properties["dsl"]; exists {
		t.Fatalf("canonical Router config mutation unexpectedly exposes DSL: %+v", configSchema)
	}
	if configSchema.AdditionalProperties != false {
		t.Fatalf("strict config decoder must publish additionalProperties=false, got %+v", configSchema)
	}

	recipePut := spec.Paths["/api/v1/config/recipes/{name}"].Put
	recipeSchema := recipePut.RequestBody.Content[string(requestBodyJSON)].Schema
	if recipeSchema == nil || recipeSchema.AdditionalProperties != false {
		t.Fatalf("strict recipe decoder must publish a closed request schema, got %+v", recipeSchema)
	}
}

func TestOpenAPISpecDerivesPathParametersFromRoutes(t *testing.T) {
	apiServer := &ClassificationAPIServer{}
	spec := apiServer.generateOpenAPISpec()

	detach := spec.Paths["/api/v1/storage/vector-stores/{id}/files/{file_id}"].Delete
	if detach == nil {
		t.Fatalf("expected vector-store file detach operation")
	}

	requireOpenAPIPathParameter(t, detach.Parameters, "id")
	requireOpenAPIPathParameter(t, detach.Parameters, "file_id")
	if strings.ContainsAny(detach.OperationID, "/{}-.") {
		t.Fatalf("expected sanitized operation ID, got %q", detach.OperationID)
	}

	listFiles := spec.Paths["/api/v1/storage/files"].Get
	if listFiles == nil {
		t.Fatalf("expected file list operation")
	}
	for _, parameter := range listFiles.Parameters {
		if parameter.In == "path" {
			t.Fatalf("expected no path parameters for /api/v1/storage/files, got %+v", listFiles.Parameters)
		}
	}
}

func TestOpenAPISpecPublishesConfigSchemaProgressiveQuery(t *testing.T) {
	server := &ClassificationAPIServer{}
	operation := server.generateOpenAPISpec().Paths[configschema.SchemaEndpoint].Get
	if operation == nil {
		t.Fatal("config schema operation is missing")
	}
	want := map[string]bool{"view": false, "path": false, "kind": false, "name": false}
	for _, parameter := range operation.Parameters {
		if _, ok := want[parameter.Name]; ok && parameter.In == "query" {
			want[parameter.Name] = true
		}
	}
	for name, present := range want {
		if !present {
			t.Errorf("config schema query parameter %q is missing", name)
		}
	}
}

func TestOpenAPISpecPublishesInvocationParameters(t *testing.T) {
	server := &ClassificationAPIServer{}
	spec := server.generateOpenAPISpec()

	eval := spec.Paths["/api/v1/routing/preview"].Post
	requireOpenAPIParameter(t, eval.Parameters, "trace", "query", false, "boolean")
	if eval.RequestBody == nil || eval.RequestBody.Content["application/json"].Schema == nil {
		t.Fatal("routing preview request schema is missing")
	}
	if got := eval.RequestBody.Content["application/json"].Schema.AdditionalProperties; got != false {
		t.Fatalf("routing preview request schema must reject unknown fields, got %#v", got)
	}
	configPatch := spec.Paths["/api/v1/config"].Patch
	requireOpenAPIParameter(t, configPatch.Parameters, "If-Match", "header", true, "string")
	configPut := spec.Paths["/api/v1/config"].Put
	requireOpenAPIParameter(t, configPut.Parameters, "If-Match", "header", true, "string")
	rollback := spec.Paths["/api/v1/config/rollback"].Post
	requireOpenAPIParameter(t, rollback.Parameters, "If-Match", "header", true, "string")
	recipe := spec.Paths["/api/v1/config/recipes/{name}"].Put
	requireOpenAPIParameter(t, recipe.Parameters, "If-Match", "header", true, "string")
	outcome := spec.Paths["/api/v1/observability/outcomes"].Post
	requireOpenAPIParameter(t, outcome.Parameters, "Idempotency-Key", "header", false, "string")
	replay := spec.Paths["/api/v1/observability/replays"].Get
	requireOpenAPIParameter(t, replay.Parameters, "cache_status", "query", false, "string")
	requireOpenAPIParameter(t, replay.Parameters, "showDetails", "query", false, "boolean")
	trajectory := spec.Paths["/api/v1/observability/replays/trajectory"].Get
	requireOpenAPIParameter(t, trajectory.Parameters, "session_id", "query", true, "string")
}

func TestOpenAPISpecPublishesProgressiveOperationQuery(t *testing.T) {
	server := &ClassificationAPIServer{}
	operation := server.generateOpenAPISpec().Paths["/openapi.json"].Get
	if operation == nil {
		t.Fatal("OpenAPI discovery operation is missing")
	}
	want := map[string]bool{
		"path": false, "method": false, "capability": false,
		"audience": false, "plane": false, "visibility": false,
	}
	for _, parameter := range operation.Parameters {
		if _, ok := want[parameter.Name]; ok && parameter.In == "query" {
			want[parameter.Name] = true
		}
	}
	for name, present := range want {
		if !present {
			t.Errorf("OpenAPI discovery query parameter %q is missing", name)
		}
	}
}

func TestOpenAPISpecPublishesRoutePolicyMetadata(t *testing.T) {
	server := &ClassificationAPIServer{}
	spec := server.generateOpenAPISpec()

	read := spec.Paths["/api/v1/config"].Get
	if read == nil || read.Permission != PermConfigRead || read.Sensitivity != SensitivitySecretView {
		t.Fatalf("config read policy metadata = %+v", read)
	}
	write := spec.Paths["/api/v1/config"].Patch
	if write == nil || write.Permission != PermConfigWrite || write.Sensitivity != SensitivityMutation || write.AuditAction != AuditActionConfigPatch {
		t.Fatalf("config patch policy metadata = %+v", write)
	}
}

func TestOpenAPISpecPublishesRuntimeBearerAuthentication(t *testing.T) {
	server := &ClassificationAPIServer{}
	spec := server.generateOpenAPISpec()

	scheme, ok := spec.Components.SecuritySchemes["bearerAuth"]
	if !ok || scheme.Type != "http" || scheme.Scheme != "bearer" {
		t.Fatalf("bearer authentication scheme = %+v, present=%v", scheme, ok)
	}
	if security := spec.Paths["/health"].Get.Security; len(security) != 0 {
		t.Fatalf("health must remain public, got security=%+v", security)
	}
	security := spec.Paths["/api/v1/config"].Get.Security
	if len(security) != 2 {
		t.Fatalf("config auth alternatives = %+v, want anonymous and bearer", security)
	}
	if _, ok := security[1]["bearerAuth"]; !ok {
		t.Fatalf("config auth alternatives = %+v, want bearerAuth", security)
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
	assertRecipeConfigOpenAPIPaths(t, spec)
	assertOpenAPIPathsAbsent(t, spec, []string{
		"/config/classification",
		"/config/system-prompts",
	})
}

func TestOpenAPISpecEndpointCanReturnOneOperation(t *testing.T) {
	apiServer := newDocumentationTestServer()
	req := httptest.NewRequest(http.MethodGet, "/openapi.json?path=%2Fapi%2Fv1%2Fconfig&method=patch", nil)
	rr := httptest.NewRecorder()

	apiServer.handleOpenAPISpec(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}
	var spec OpenAPISpec
	if err := json.Unmarshal(rr.Body.Bytes(), &spec); err != nil {
		t.Fatalf("failed to unmarshal filtered OpenAPI spec: %v", err)
	}
	if len(spec.Paths) != 1 {
		t.Fatalf("expected one selected path, got %d", len(spec.Paths))
	}
	selected := spec.Paths["/api/v1/config"]
	if selected.Patch == nil {
		t.Fatal("expected selected PATCH operation")
	}
	if selected.Get != nil || selected.Put != nil {
		t.Fatalf("expected only PATCH, got %+v", selected)
	}
}

func TestOpenAPISpecEndpointCanFilterForAgentConfigOperations(t *testing.T) {
	apiServer := newDocumentationTestServer()
	req := httptest.NewRequest(http.MethodGet, "/openapi.json?capability=config&audience=agent", nil)
	rr := httptest.NewRecorder()

	apiServer.handleOpenAPISpec(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d: %s", rr.Code, rr.Body.String())
	}
	var spec OpenAPISpec
	if err := json.Unmarshal(rr.Body.Bytes(), &spec); err != nil {
		t.Fatalf("failed to unmarshal filtered OpenAPI spec: %v", err)
	}
	if len(spec.Paths) == 0 {
		t.Fatal("expected filtered config operations")
	}
	for path, operations := range spec.Paths {
		for _, operation := range []*OpenAPIOperation{operations.Get, operations.Post, operations.Patch, operations.Put, operations.Delete} {
			if operation == nil {
				continue
			}
			if !slices.Contains(operation.Tags, "config") || !slices.Contains(operation.Audiences, APIAudienceAgent) {
				t.Fatalf("operation %s has wrong semantic metadata: %+v", path, operation)
			}
		}
	}
}

func TestOpenAPISpecEndpointRejectsUnknownSelection(t *testing.T) {
	apiServer := newDocumentationTestServer()
	tests := []struct {
		name       string
		target     string
		statusCode int
		errorCode  string
	}{
		{name: "method without path", target: "/openapi.json?method=GET", statusCode: http.StatusBadRequest, errorCode: "INVALID_OPENAPI_FILTER"},
		{name: "unknown path", target: "/openapi.json?path=%2Fmissing", statusCode: http.StatusNotFound, errorCode: "OPENAPI_PATH_NOT_FOUND"},
		{name: "unknown operation", target: "/openapi.json?path=%2Fhealth&method=POST", statusCode: http.StatusNotFound, errorCode: "OPENAPI_OPERATION_NOT_FOUND"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, test.target, nil)
			rr := httptest.NewRecorder()
			apiServer.handleOpenAPISpec(rr, req)
			if rr.Code != test.statusCode {
				t.Fatalf("expected %d, got %d: %s", test.statusCode, rr.Code, rr.Body.String())
			}
			if code := parseErrorResponse(t, rr.Body.Bytes()); code != test.errorCode {
				t.Fatalf("expected error code %q, got %q", test.errorCode, code)
			}
		})
	}
}

func assertRecipeConfigOpenAPIPaths(t *testing.T, spec OpenAPISpec) {
	t.Helper()

	collection := spec.Paths["/api/v1/config/recipes"]
	if collection.Get == nil {
		t.Fatal("expected /api/v1/config/recipes GET to be documented")
	}
	validation := spec.Paths["/api/v1/config/recipes/validate"]
	if validation.Post == nil || validation.Post.RequestBody == nil {
		t.Fatal("expected recipe validation POST body to be documented")
	}
	item := spec.Paths["/api/v1/config/recipes/{name}"]
	if item.Get == nil || item.Put == nil || item.Delete == nil {
		t.Fatalf("expected recipe item GET, PUT, and DELETE operations, got %+v", item)
	}
	requireOpenAPIPathParameter(t, item.Put.Parameters, "name")
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
	if len(spec.Components.SecuritySchemes) == 0 {
		t.Error("expected OpenAPI authentication schemes")
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

	routerPath, exists := spec.Paths["/api/v1/config"]
	if !exists {
		t.Fatalf("expected /api/v1/config to be documented in OpenAPI spec")
	}
	if routerPath.Patch == nil || routerPath.Put == nil || routerPath.Get == nil {
		t.Fatalf("expected /api/v1/config to document GET, PATCH, and PUT, got %+v", routerPath)
	}
	if _, ok := routerPath.Patch.Responses["413"]; !ok {
		t.Fatalf("expected /api/v1/config PATCH to document 413 request body limit response")
	}
	if routerPath.Patch.RequestBody == nil || routerPath.Patch.RequestBody.Description == "" {
		t.Fatalf("expected /api/v1/config PATCH to document request body constraints")
	}
}

func requireOpenAPIPathParameter(t *testing.T, parameters []OpenAPIParameter, name string) {
	t.Helper()
	requireOpenAPIParameter(t, parameters, name, "path", true, "string")
}

func requireOpenAPIParameter(
	t *testing.T,
	parameters []OpenAPIParameter,
	name, location string,
	required bool,
	valueType string,
) {
	t.Helper()

	for _, parameter := range parameters {
		if parameter.Name != name {
			continue
		}
		if parameter.In != location {
			t.Fatalf("expected %q parameter location %s, got %q", name, location, parameter.In)
		}
		if parameter.Required != required {
			t.Fatalf("expected %q required=%v, got %v", name, required, parameter.Required)
		}
		if parameter.Schema.Type != valueType {
			t.Fatalf("expected %q parameter schema %s, got %+v", name, valueType, parameter.Schema)
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
		"/api/v1/diagnostics/classify/batch",
		"/api/v1/routing/preview",
		"/api/v1/diagnostics/nli",
		"/api/v1/diagnostics/embeddings",
		"/api/v1/diagnostics/similarity/batch",
		"/openapi.json",
		"/docs",
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
