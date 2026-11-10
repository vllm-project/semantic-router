package auth

import (
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestPolicyMuxHeadLookup(t *testing.T) {
	get := ReadPolicy(http.MethodGet, PermConfigRead, SensitivitySensitive, ResourceOwnerConfig)
	get.AdditionalPermissions = []string{PermReplayRead}
	explicit := ReadPolicy(http.MethodHead, PermUsersManage, SensitivitySecret, ResourceOwnerAuth)
	for _, tc := range []struct {
		name     string
		policies []RoutePolicy
		want     RoutePolicy
		lookup   RouteLookup
	}{
		{"inherits complete GET policy", []RoutePolicy{get}, get, RouteFound},
		{"explicit HEAD after GET", []RoutePolicy{get, explicit}, explicit, RouteFound},
		{"explicit HEAD before GET", []RoutePolicy{explicit, get}, explicit, RouteFound},
		{"POST only", []RoutePolicy{PublicPolicy(http.MethodPost)}, RoutePolicy{}, RouteMethodNotAllowed},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mux := NewPolicyMux()
			mux.HandlePolicyFunc(Route("/api/items/{id}", tc.policies...), func(http.ResponseWriter, *http.Request) {})
			got, lookup := mux.LookupRoutePolicy(http.MethodHead, "/api/items/a%2Fb")
			if lookup != tc.lookup || !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("HEAD lookup = %+v, %v; want %+v, %v", got, lookup, tc.want, tc.lookup)
			}
		})
	}
}

func TestPolicyMuxHeadHTTP(t *testing.T) {
	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "head-policy@example.com", RoleRead, "active")
	for _, configured := range []bool{true, false} {
		t.Run(map[bool]string{true: "configured", false: "unavailable"}[configured], func(t *testing.T) {
			mux := NewPolicyMux()
			serve := func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodHead {
					t.Errorf("handler received %s, want HEAD", r.Method)
				}
				w.Header().Set("X-Head-Handler", "reached")
				_, _ = io.WriteString(w, "healthy")
			}
			mux.HandlePolicyFunc(PublicRoute("/healthz", http.MethodGet), serve)
			mux.HandlePolicyFunc(ProtectedRoute("/api/config", PermConfigRead, SensitivityOperational, ResourceOwnerConfig, http.MethodGet), serve)
			mux.HandlePolicyFunc(ProtectedRoute("/api/users", PermUsersManage, SensitivitySensitive, ResourceOwnerAuth, http.MethodGet), serve)
			mux.HandlePolicyFunc(PublicRoute("/submit", http.MethodPost), serve)
			mux.Seal()
			handler := unavailableWithRoutePolicy(mux)(mux)
			if configured {
				handler = AuthenticateRequest(svc, mux)(mux)
			}
			server := httptest.NewServer(handler)
			defer server.Close()
			for _, tc := range []struct {
				path          string
				authenticated bool
				status        int
			}{
				{"/healthz", false, http.StatusOK},
				{"/api/config", false, map[bool]int{true: http.StatusUnauthorized, false: http.StatusServiceUnavailable}[configured]},
				{"/api/config", true, map[bool]int{true: http.StatusOK, false: http.StatusServiceUnavailable}[configured]},
				{"/api/users", true, map[bool]int{true: http.StatusForbidden, false: http.StatusServiceUnavailable}[configured]},
				{"/submit", false, http.StatusMethodNotAllowed},
			} {
				var request *http.Request
				if tc.authenticated {
					request = newAuthenticatedRequest(t, svc, user, http.MethodHead, server.URL+tc.path, "")
					request.RequestURI = ""
				} else {
					var err error
					request, err = http.NewRequest(http.MethodHead, server.URL+tc.path, nil)
					if err != nil {
						t.Fatal(err)
					}
				}
				response, err := server.Client().Do(request)
				if err != nil {
					t.Fatal(err)
				}
				body, err := io.ReadAll(response.Body)
				response.Body.Close()
				if err != nil {
					t.Fatal(err)
				}
				if response.StatusCode != tc.status {
					t.Errorf("HEAD %s (authenticated=%v) = %d; want %d", tc.path, tc.authenticated, response.StatusCode, tc.status)
				}
				if len(body) != 0 {
					t.Errorf("HEAD %s returned body %q", tc.path, body)
				}
				reached := response.Header.Get("X-Head-Handler") != ""
				if reached != (tc.status == http.StatusOK) {
					t.Errorf("HEAD %s handler reached = %v", tc.path, reached)
				}
			}
		})
	}
}
