package router

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/routercontract"
)

func TestManagementGatewaySharesRBACAndReadonlyPolicy(t *testing.T) {
	calls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if r.Header.Get("Authorization") != "Bearer router-management" || r.Header.Get("Cookie") != "" || r.URL.Query().Get("authToken") != "" {
			t.Errorf("browser identity reached Router: %s %v", r.URL.String(), r.Header)
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer upstream.Close()
	for _, readonly := range []bool{false, true} {
		mux := http.NewServeMux()
		registerRouterAPIProxy(mux, &config.Config{RouterAPIURL: upstream.URL, ReadonlyMode: readonly}, nil, routerProxyCredentialProvider{token: "router-management"})
		for _, policy := range routercontract.ManagementPolicies() {
			path := strings.NewReplacer("{type}", "rag", "{id}", "record-1", "{name}", "example").Replace(policy.Path)
			perms := auth.RequiredPermissions(policy.Method, path)
			if !reflect.DeepEqual(perms, policy.Permissions) {
				t.Fatalf("RBAC drift for %s %s: %v", policy.Method, path, perms)
			}
			request := httptest.NewRequest(policy.Method, path+"?authToken=browser-query", strings.NewReader(`{}`))
			request.Header.Set("Authorization", "Bearer browser-jwt")
			request.Header.Set("Cookie", "vsr_session=browser-cookie")
			response := httptest.NewRecorder()
			before := calls
			mux.ServeHTTP(response, request)
			want := http.StatusNoContent
			if readonly && policy.Mutation {
				want = http.StatusForbidden
			}
			if response.Code != want {
				t.Fatalf("readonly=%v %s %s = %d want %d: %s", readonly, policy.Method, path, response.Code, want, response.Body.String())
			}
			if (calls == before) != (readonly && policy.Mutation) {
				t.Fatalf("readonly mutation forwarding mismatch: %+v", policy)
			}
		}
	}
}

func TestManagementGatewayRejectsUndeclaredAndOldRoutes(t *testing.T) {
	calls := 0
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls++; w.WriteHeader(http.StatusNoContent) }))
	defer upstream.Close()
	mux := http.NewServeMux()
	registerRouterAPIProxy(mux, &config.Config{RouterAPIURL: upstream.URL}, nil, routerProxyCredentialProvider{token: "managed"})
	for _, test := range []struct {
		method, path string
		want         int
	}{
		{http.MethodGet, "/api/router/api/v1/response-cache/stats", http.StatusNotFound},
		{http.MethodPost, "/api/router/api/v1/context-compression/preview", http.StatusForbidden},
		{http.MethodPatch, "/api/router/api/v1/config", http.StatusForbidden},
		{http.MethodPut, "/api/router/api/v1/config/recipes/private", http.StatusForbidden},
		{http.MethodGet, "/api/router/api/v1/observability/replays/id/extra", http.StatusNotFound},
		{http.MethodPost, "/api/router/api/v1/plugins/unknown/probe", http.StatusForbidden},
		{http.MethodPost, "/api/router/api/v1/diagnostics/models/unknown", http.StatusForbidden},
	} {
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, httptest.NewRequest(test.method, test.path, nil))
		if w.Code != test.want {
			t.Fatalf("%+v got %d", test, w.Code)
		}
	}
	if calls != 0 {
		t.Fatalf("undeclared route reached Router %d times", calls)
	}
}
