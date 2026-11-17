package router

import (
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/apicontract"
	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/handlers"
)

func TestCapabilitiesRouteRequiresSession(t *testing.T) {
	server := setupRouteInventoryServer(t)
	policy, lookup := server.routePolicies.LookupRoutePolicy(http.MethodGet, dashboardCapabilitiesPath)
	if lookup != auth.RouteFound || policy.Permission != auth.PermSessionRead {
		t.Fatalf("policy = %+v lookup = %v", policy, lookup)
	}
	for _, role := range auth.SupportedRoles {
		if !slices.Contains(auth.DefaultRolePermissions[role], auth.PermSessionRead) {
			t.Errorf("role %s cannot read capabilities", role)
		}
	}
	recorder := httptest.NewRecorder()
	server.Handler.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, dashboardCapabilitiesPath, nil))
	if recorder.Code != http.StatusUnauthorized {
		t.Fatalf("anonymous status = %d, want %d", recorder.Code, http.StatusUnauthorized)
	}
}

func TestCapabilitiesOperationEnumeratesStatesAndReasons(t *testing.T) {
	server := setupRouteInventoryServer(t)
	spec, err := dashboardOpenAPISpec(server.routePolicies.Contracts())
	if err != nil {
		t.Fatal(err)
	}
	operation := spec.Paths[dashboardCapabilitiesPath]["get"]
	if operation == nil || operation.SchemaStatus != apicontract.SchemaTyped || operation.OperationID != "listCapabilities" {
		t.Fatalf("operation = %+v", operation)
	}
	items := operation.Responses["200"].Content["application/json"].Schema.Properties["capabilities"].Items
	if items == nil {
		t.Fatal("capabilities schema has no items")
	}
	if got := items.Properties["reason"].Enum; !slices.Equal(got, handlers.CapabilityReasons) {
		t.Errorf("reason enum = %v, want %v", got, handlers.CapabilityReasons)
	}
	if got := items.Properties["state"].Enum; len(got) != len(handlers.CapabilityStates) {
		t.Errorf("state enum = %v", got)
	}
	for _, field := range []string{"id", "enabled", "configured", "reachable", "authorized", "degraded", "state"} {
		if !slices.Contains(items.Required, field) {
			t.Errorf("capability field %q is not required", field)
		}
	}
}
