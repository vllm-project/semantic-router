package routercontract

import (
	"net/http"
	"reflect"
	"strings"
	"testing"
)

func TestManagementPolicyMatchesOnlyCompleteDeclaredMethodPaths(t *testing.T) {
	seen := map[string]bool{}
	for _, policy := range ManagementPolicies() {
		key := policy.Method + " " + policy.Path
		if seen[key] || len(policy.Permissions) == 0 {
			t.Fatalf("invalid policy: %+v", policy)
		}
		seen[key] = true
		if policy.Mutation && (policy.Method == http.MethodGet || policy.Method == http.MethodHead) {
			t.Fatalf("read mutation: %+v", policy)
		}
		path := strings.NewReplacer("{type}", "rag", "{id}", "record-1", "{name}", "example").Replace(policy.Path)
		got, ok := LookupManagement(policy.Method, path)
		if !ok || !reflect.DeepEqual(got.Permissions, policy.Permissions) || got.Mutation != policy.Mutation {
			t.Fatalf("policy not addressable: %+v", policy)
		}
		if _, ok := LookupManagement(policy.Method, path+"/unexpected/extra/depth"); ok {
			t.Fatalf("prefix route leaked: %s", path)
		}
	}
	for _, test := range []struct{ method, path string }{
		{http.MethodPut, "/api/router/api/v1/config"},
		{http.MethodPatch, "/api/router/api/v1/config"},
		{http.MethodPost, "/api/router/api/v1/config/rollback"},
		{http.MethodGet, "/api/router/api/v1/response-cache/stats"},
		{http.MethodPost, "/api/router/api/v1/context-compression/preview"},
		{http.MethodDelete, "/api/router/api/v1/plugins/rag/bindings"},
		{http.MethodGet, "/api/router/api/v1/plugins/../bindings"},
		{http.MethodGet, "/api/router/api/v1/plugins/%2e%2e/bindings"},
		{http.MethodGet, "/api/router/api/v1/plugins//bindings"},
		{http.MethodGet, "/api/router/api/v1/observability/replays/record/extra"},
		{http.MethodGet, "/api/router/api/v1/plugins/rag/bindings/"},
		{http.MethodPost, "/api/router/api/v1/diagnostics/models/unknown"},
		{http.MethodGet, "/api/router/API/v1/plugins"},
	} {
		if _, ok := LookupManagement(test.method, test.path); ok {
			t.Fatalf("undeclared route allowed: %+v", test)
		}
	}
}
