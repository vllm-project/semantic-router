package router

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/apicontract"
	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

const (
	dashboardOpenAPIArtifact  = "dashboard.openapi.json"
	updateDashboardOpenAPIEnv = "UPDATE_DASHBOARD_OPENAPI"
)

// The committed artifact is rendered from the full route inventory (MCP,
// OpenClaw, and ML pipeline enabled). Regenerate it with
// `make dashboard-openapi-generate`.
func TestDashboardOpenAPIArtifactIsCurrent(t *testing.T) {
	server := setupRouteInventoryServer(t)
	spec, err := dashboardOpenAPISpec(server.routePolicies.Contracts())
	if err != nil {
		t.Fatal(err)
	}
	rendered, err := json.MarshalIndent(spec, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	rendered = append(rendered, '\n')

	if os.Getenv(updateDashboardOpenAPIEnv) == "1" {
		if writeErr := os.WriteFile(dashboardOpenAPIArtifact, rendered, 0o644); writeErr != nil {
			t.Fatal(writeErr)
		}
		return
	}
	committed, err := os.ReadFile(dashboardOpenAPIArtifact)
	if err != nil {
		t.Fatalf("read %s: %v; run make dashboard-openapi-generate", dashboardOpenAPIArtifact, err)
	}
	if !bytes.Equal(committed, rendered) {
		t.Fatalf("%s is out of date with the route registration; run make dashboard-openapi-generate", dashboardOpenAPIArtifact)
	}
}

func TestDashboardOpenAPICoversEveryRegisteredRoute(t *testing.T) {
	server := setupRouteInventoryServer(t)
	contracts := server.routePolicies.Contracts()
	spec, err := dashboardOpenAPISpec(contracts)
	if err != nil {
		t.Fatal(err)
	}

	rendered := map[string]bool{}
	for path, item := range spec.Paths {
		for method, operation := range item {
			pattern := operation.RoutePattern
			if pattern == "" {
				pattern = path
			}
			rendered[strings.ToUpper(method)+" "+pattern] = true
		}
	}

	usedExemptions := map[string]bool{}
	for _, contract := range contracts {
		for prefix := range openAPIExemptPrefixes {
			if strings.HasPrefix(contract.Pattern, prefix) {
				usedExemptions[prefix] = true
			}
		}
		if _, exempt := openAPIExemption(contract.Pattern); exempt {
			continue
		}
		for _, policy := range contract.Policies {
			if !rendered[policy.Method+" "+contract.Pattern] {
				t.Errorf("%s %s is registered but absent from the OpenAPI document", policy.Method, contract.Pattern)
			}
		}
	}
	for prefix := range openAPIExemptPrefixes {
		if !usedExemptions[prefix] {
			t.Errorf("OpenAPI exemption %q matches no registered route; remove it", prefix)
		}
	}
}

func TestDashboardOpenAPIWriteRoutesArePolicyComplete(t *testing.T) {
	server := setupRouteInventoryServer(t)
	spec, err := dashboardOpenAPISpec(server.routePolicies.Contracts())
	if err != nil {
		t.Fatal(err)
	}
	for path, item := range spec.Paths {
		for method, operation := range item {
			if method == "get" || method == "head" || len(operation.Security) == 0 {
				continue
			}
			if !operation.CSRF {
				t.Errorf("%s %s accepts cookie writes without CSRF", method, path)
			}
			if operation.MaxBodyBytes <= 0 && !operation.ProxyUpstream {
				t.Errorf("%s %s has no request body bound", method, path)
			}
			if operation.Permission == "" {
				t.Errorf("%s %s has no permission", method, path)
			}
			// Read-only POSTs such as previews are deliberately unaudited;
			// every audited write must name its action.
			if operation.AuditMode != string(auth.AuditNone) && operation.AuditAction == "" {
				t.Errorf("%s %s is audited without an audit action", method, path)
			}
		}
	}
}

func TestDashboardOpenAPIDescribesFirstRunOperations(t *testing.T) {
	server := setupRouteInventoryServer(t)
	spec, err := dashboardOpenAPISpec(server.routePolicies.Contracts())
	if err != nil {
		t.Fatal(err)
	}
	for _, typed := range []struct {
		path, method, operationID string
		request                   bool
	}{
		{"/healthz", "get", "getHealthz", false},
		{"/openapi.json", "get", "getDashboardOpenAPI", false},
		{"/api/settings", "get", "getSettings", false},
		{"/api/setup/state", "get", "getSetupState", false},
		{"/api/setup/import-remote", "post", "importRemoteSetupConfig", true},
		{"/api/setup/validate", "post", "validateSetupConfig", true},
		{"/api/setup/activate", "post", "activateSetupConfig", true},
		{"/api/setup/presets", "get", "listSetupPresets", false},
		{"/api/setup/presets/delta", "post", "computeSetupPresetDelta", true},
	} {
		operation := spec.Paths[typed.path][typed.method]
		if operation == nil {
			t.Errorf("%s %s is missing", typed.method, typed.path)
			continue
		}
		if operation.SchemaStatus != apicontract.SchemaTyped || operation.OperationID != typed.operationID {
			t.Errorf("%s %s = status %q id %q", typed.method, typed.path, operation.SchemaStatus, operation.OperationID)
		}
		success := operation.Responses["200"].Content["application/json"].Schema
		if success == nil || (success.Type != "object" && success.Type != "array") {
			t.Errorf("%s %s has no concrete success schema", typed.method, typed.path)
		}
		if typed.request && (operation.RequestBody == nil || operation.RequestBody.Content["application/json"].Schema == nil) {
			t.Errorf("%s %s has no request schema", typed.method, typed.path)
		}
	}
	settings := spec.Paths["/api/settings"]["get"].Responses["200"].Content["application/json"].Schema
	if _, ok := settings.Properties["readonlyMode"]; !ok {
		t.Errorf("settings schema omits readonlyMode: %+v", settings.Properties)
	}
}

func TestDashboardOpenAPIRouteRequiresAuthentication(t *testing.T) {
	server := setupRouteInventoryServer(t)
	policy, lookup := server.routePolicies.LookupRoutePolicy(http.MethodGet, dashboardOpenAPIPath)
	if lookup != auth.RouteFound || policy.Permission != auth.PermConfigRead {
		t.Fatalf("policy = %+v lookup = %v", policy, lookup)
	}
	recorder := httptest.NewRecorder()
	server.Handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, dashboardOpenAPIPath, nil))
	if recorder.Code != http.StatusUnauthorized {
		t.Fatalf("anonymous status = %d, want %d", recorder.Code, http.StatusUnauthorized)
	}
}

type staticContracts []auth.RouteContract

func (contracts staticContracts) Contracts() []auth.RouteContract { return contracts }

func TestDashboardOpenAPIServesRegisteredRoutes(t *testing.T) {
	mux := http.NewServeMux()
	registerOpenAPIRoute(mux, staticContracts{
		auth.PublicRoute("/healthz", http.MethodGet),
		auth.ProtectedRoute("/embedded/grafana/", auth.PermLogsRead, auth.SensitivitySensitive, auth.ResourceOwnerObservability, http.MethodGet),
	})
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, dashboardOpenAPIPath, nil))
	if recorder.Code != http.StatusOK || recorder.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("status = %d content-type = %q", recorder.Code, recorder.Header().Get("Content-Type"))
	}
	var spec apicontract.Spec
	if err := json.Unmarshal(recorder.Body.Bytes(), &spec); err != nil {
		t.Fatal(err)
	}
	if spec.Paths["/healthz"]["get"] == nil {
		t.Errorf("served document omits /healthz: %v", spec.Paths)
	}
	for path := range spec.Paths {
		if strings.HasPrefix(path, "/embedded/") {
			t.Errorf("served document includes exempt path %s", path)
		}
	}
}

func TestDashboardOpenAPIReportsRenderFailure(t *testing.T) {
	mux := http.NewServeMux()
	registerOpenAPIRoute(mux, staticContracts{
		auth.ProtectedRoute("/api/rooms/{id}", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet),
		auth.ProtectedRoute("/api/rooms/", auth.PermConfigRead, auth.SensitivityOperational, auth.ResourceOwnerConfig, http.MethodGet),
	})
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, dashboardOpenAPIPath, nil))
	if recorder.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusInternalServerError)
	}
}

// The activate handler encodes its read-only body from a map, so compare its
// keys with the documented schema.
func TestSetupActivateReadonlyBodyMatchesOpenAPI(t *testing.T) {
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	handler := handlers.SetupActivateHandler(configPath, true, t.TempDir(), setupmode.New(configPath, false))
	recorder := httptest.NewRecorder()
	handler(recorder, httptest.NewRequest(http.MethodPost, "/api/setup/activate", strings.NewReader(`{}`)))
	if recorder.Code != http.StatusForbidden || recorder.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("status = %d content-type = %q", recorder.Code, recorder.Header().Get("Content-Type"))
	}
	var body map[string]any
	if err := json.Unmarshal(recorder.Body.Bytes(), &body); err != nil {
		t.Fatal(err)
	}
	schema := apicontract.SchemaFor[setupErrorBody]()
	if len(body) != len(schema.Properties) {
		t.Fatalf("body keys %v, documented %v", body, schema.Properties)
	}
	for key := range body {
		if _, ok := schema.Properties[key]; !ok {
			t.Errorf("undocumented key %q", key)
		}
	}
	if body["error"] != "readonly_mode" {
		t.Errorf("error = %v, want readonly_mode", body["error"])
	}
}

// The handler reports failure stages as string literals, so read them from
// its source.
func TestSetupActivationFailureStagesMatchHandler(t *testing.T) {
	source, err := os.ReadFile("../handlers/setup.go")
	if err != nil {
		t.Fatal(err)
	}
	matches := regexp.MustCompile(`failSetupActivation\([^)]*"([a-z_]+)"\)`).FindAllSubmatch(source, -1)
	reported := make([]string, 0, len(matches))
	for _, match := range matches {
		reported = append(reported, string(match[1]))
	}
	if !slices.Equal(reported, setupActivationFailureStages) {
		t.Fatalf("handler reports stages %v, documented %v", reported, setupActivationFailureStages)
	}
}

// jsonLiteralKeys reads the quoted keys of the map literal enclosing anchor,
// the exact text of one of its entries. It grounds a schema-exclusivity test
// in the real field names a handler encodes, not a hand-copied list.
func jsonLiteralKeys(t *testing.T, path, anchor string) []string {
	t.Helper()
	source, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	start := bytes.Index(source, []byte(anchor))
	if start < 0 {
		t.Fatalf("anchor %q not found in %s", anchor, path)
	}
	open := bytes.LastIndexByte(source[:start], '{')
	closeOffset := bytes.IndexByte(source[start:], '}')
	if open < 0 || closeOffset < 0 {
		t.Fatalf("could not bound the map literal around %q in %s", anchor, path)
	}
	block := source[open : start+closeOffset]
	matches := regexp.MustCompile(`"(\w+)":`).FindAllSubmatch(block, -1)
	keys := make([]string, 0, len(matches))
	for _, match := range matches {
		keys = append(keys, string(match[1]))
	}
	return keys
}

// The 500 oneOf only validates real responses if its two branches never both
// accept the same body. Each body's field set comes from the handler source,
// not a hand-copied list, so a renamed or added field fails this test.
func TestSetupActivationErrorSchemasAreExclusive(t *testing.T) {
	activationKeys := jsonLiteralKeys(t, "../handlers/setup_activation_failure.go", `"error":                  "setup_activation_failed"`)
	coordinationKeys := jsonLiteralKeys(t, "../handlers/runtime_config_coordinator.go", `"error":   mutationErr.code`)

	activationSchema := setupActivationFailureSchema()
	coordinationSchema := setupConfigCoordinationFailureSchema()

	if !activationSchema.Accepts(activationKeys) {
		t.Errorf("activation failure body %v does not match its own schema", activationKeys)
	}
	if coordinationSchema.Accepts(activationKeys) {
		t.Errorf("activation failure body %v also matches the coordination schema; oneOf is ambiguous", activationKeys)
	}
	if !coordinationSchema.Accepts(coordinationKeys) {
		t.Errorf("coordination failure body %v does not match its own schema", coordinationKeys)
	}
	if activationSchema.Accepts(coordinationKeys) {
		t.Errorf("coordination failure body %v also matches the activation schema; oneOf is ambiguous", coordinationKeys)
	}
}
