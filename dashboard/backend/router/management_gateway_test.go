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
		mux := auth.NewPolicyMux()
		registerRouterAPIProxy(mux, &config.Config{RouterAPIURL: upstream.URL, ReadonlyMode: readonly}, nil, nil, routerProxyCredentialProvider{token: "router-management"})
		for _, policy := range routercontract.ManagementPolicies() {
			if isKnowledgeBasePath(policy.Path) {
				// Knowledge-base storage is bound to the classifier proxy in
				// registerConfigRoutes rather than the generic gateway.
				continue
			}
			path := strings.NewReplacer("{type}", "rag", "{id}", "record-1", "{name}", "example").Replace(policy.Path)
			routePolicy, lookup := mux.LookupRoutePolicy(policy.Method, path)
			if lookup != auth.RouteFound || !reflect.DeepEqual(routePolicy.Permissions, policy.Permissions) {
				t.Fatalf("RBAC drift for %s %s: lookup=%v permissions=%v", policy.Method, path, lookup, routePolicy.Permissions)
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
	mux := auth.NewPolicyMux()
	registerRouterAPIProxy(mux, &config.Config{RouterAPIURL: upstream.URL}, nil, nil, routerProxyCredentialProvider{token: "managed"})
	// Undeclared routes never receive a contract, so the mux itself cannot
	// dispatch them and the authentication layer denies them ahead of it.
	for _, test := range []struct {
		method, path string
		want         auth.RouteLookup
	}{
		{http.MethodGet, "/api/router/api/v1/response-cache/stats", auth.RouteNotFound},
		{http.MethodPost, "/api/router/api/v1/context-compression/preview", auth.RouteNotFound},
		{http.MethodPatch, "/api/router/api/v1/config", auth.RouteNotFound},
		{http.MethodPut, "/api/router/api/v1/config/recipes/private", auth.RouteNotFound},
		{http.MethodGet, "/api/router/api/v1/observability/replays/id/extra", auth.RouteNotFound},
		{http.MethodPost, "/api/router/api/v1/plugins/unknown/probe", auth.RouteNotFound},
		{http.MethodPost, "/api/router/api/v1/diagnostics/models/unknown", auth.RouteNotFound},
		{http.MethodPost, "/api/router/api/v1/config/hash", auth.RouteMethodNotAllowed},
	} {
		if _, lookup := mux.LookupRoutePolicy(test.method, test.path); lookup != test.want {
			t.Fatalf("%+v lookup = %v", test, lookup)
		}
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, httptest.NewRequest(test.method, test.path, nil))
		wantStatus := http.StatusNotFound
		if test.want == auth.RouteMethodNotAllowed {
			// The mux still owns the pattern; the gateway allowlist refuses the
			// method itself, and the authentication layer answers 405 first.
			wantStatus = http.StatusForbidden
		}
		if w.Code != wantStatus {
			t.Fatalf("%+v dispatched with status %d, want %d", test, w.Code, wantStatus)
		}
	}
	if calls != 0 {
		t.Fatalf("undeclared route reached Router %d times", calls)
	}
}
