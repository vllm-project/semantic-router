package router

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/apicontract"
	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

func TestDashboardOpenAPIDescribesNativeRouteDiagnostics(t *testing.T) {
	server := setupRouteInventoryServer(t)
	spec, err := dashboardOpenAPISpec(server.routePolicies.Contracts())
	if err != nil {
		t.Fatal(err)
	}
	operations := spec.Paths["/api/decision-model/routes"]
	for _, method := range []string{"get", "post"} {
		operation := operations[method]
		if operation == nil || operation.SchemaStatus != apicontract.SchemaTyped {
			t.Fatalf("%s native diagnostics has no documented contract", method)
		}
		if len(operation.Security) != 2 {
			t.Errorf("%s native diagnostics lost Dashboard authentication", method)
		}
		for _, status := range []string{"401", "403"} {
			if len(operation.Responses[status].Content) == 0 {
				t.Errorf("%s lacks authentication response %s", method, status)
			}
		}
	}
	get, post := operations["get"], operations["post"]
	if get.Permission != auth.PermConfigRead || get.CSRF || get.RequestBody != nil {
		t.Errorf("discovery policy changed: %+v", get)
	}
	if post.Permission != auth.PermEvalRun || !post.CSRF || post.MaxBodyBytes != 2<<20 {
		t.Errorf("execution policy changed: %+v", post)
	}
	if post.RequestBody == nil || !post.RequestBody.Required {
		t.Fatal("execution needs a required request wrapper")
	}
	schema := post.RequestBody.Content["application/json"].Schema
	if schema == nil || !schema.Accepts([]string{"model", "request"}) || schema.Accepts([]string{"model"}) || schema.Accepts([]string{"model", "request", "url"}) {
		t.Fatalf("wrapper must require model and request, and reject extra fields: %+v", schema)
	}
	request := schema.Properties["request"]
	if request.Type != "object" || request.Nullable || request.AdditionalProperties != true {
		t.Errorf("native request should remain an open, non-null JSON object: %+v", request)
	}
}

func TestNativeRouteDiagnosticsRequireDashboardAuthentication(t *testing.T) {
	server := setupRouteInventoryServer(t)
	for _, method := range []string{http.MethodGet, http.MethodPost} {
		recorder := httptest.NewRecorder()
		request := httptest.NewRequest(method, "/api/decision-model/routes", strings.NewReader(`{"model":"vllm-sr/auto","request":{}}`))
		server.Handler.ServeHTTP(recorder, request)
		if recorder.Code != http.StatusUnauthorized {
			t.Errorf("anonymous %s status = %d, want 401", method, recorder.Code)
		}
	}
}
