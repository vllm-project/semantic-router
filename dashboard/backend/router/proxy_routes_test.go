package router

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

type routerProxyCredentialProvider struct {
	token string
}

func TestRegisterProxyRoutesDoesNotExposeFleetSimAPI(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	registerProxyRoutes(mux, &config.Config{})

	req := httptest.NewRequest(http.MethodGet, "/api/fleet-sim/api/workloads", nil)
	_, pattern := mux.Handler(req)
	if pattern != "/api/" {
		t.Fatalf("matched route = %q, want generic API fallback %q", pattern, "/api/")
	}

	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, req)
	if recorder.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusBadGateway)
	}
	if !strings.Contains(recorder.Body.String(), "No API handler configured for this path") {
		t.Fatalf("response body = %q, want generic API fallback", recorder.Body.String())
	}
}

func (provider routerProxyCredentialProvider) ManagementCredential() (string, error) {
	return provider.token, nil
}

func TestRouterAPIProxyReplacesBrowserAuthorization(t *testing.T) {
	var authorization string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorization = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	mux := http.NewServeMux()
	registerRouterAPIProxy(
		mux,
		&config.Config{RouterAPIURL: server.URL},
		nil,
		routerProxyCredentialProvider{token: "router-service-token"},
	)
	req := httptest.NewRequest(http.MethodGet, "/api/router/v1/models", nil)
	req.Header.Set("Authorization", "Bearer dashboard-user-jwt")
	recorder := httptest.NewRecorder()

	mux.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusNoContent {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusNoContent)
	}
	if authorization != "Bearer router-service-token" {
		t.Fatalf("Authorization = %q", authorization)
	}
}

func TestRouterOutcomeProxyUsesServiceCredential(t *testing.T) {
	var authorization string
	var proxyAuthorization string
	var cookie string
	var queryCredential string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorization = r.Header.Get("Authorization")
		proxyAuthorization = r.Header.Get("Proxy-Authorization")
		cookie = r.Header.Get("Cookie")
		queryCredential = r.URL.Query().Get("authToken")
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	mux := http.NewServeMux()
	registerRouterAPIProxy(
		mux,
		&config.Config{RouterAPIURL: server.URL},
		nil,
		routerProxyCredentialProvider{token: "router-service-token"},
	)
	req := httptest.NewRequest(http.MethodPost, "/api/router/v1/router/outcomes?authToken=query-user-jwt", nil)
	req.Header.Set("Authorization", "Bearer dashboard-feedback-user-jwt")
	req.Header.Set("Proxy-Authorization", "Bearer proxy-user-jwt")
	req.Header.Set("Cookie", "vsr_session=cookie-user-jwt")
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusNoContent {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusNoContent)
	}
	if authorization != "Bearer router-service-token" {
		t.Fatalf("Authorization = %q", authorization)
	}
	if proxyAuthorization != "" || cookie != "" || queryCredential != "" {
		t.Fatalf("browser credentials leaked: proxy=%q cookie=%q query=%q", proxyAuthorization, cookie, queryCredential)
	}
}

func TestRouterAPIProxyRejectsUnknownManagementMutation(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	mux := http.NewServeMux()
	registerRouterAPIProxy(
		mux,
		&config.Config{RouterAPIURL: server.URL},
		nil,
		routerProxyCredentialProvider{token: "router-service-token"},
	)
	req := httptest.NewRequest(http.MethodPost, "/api/router/v1/unknown-mutation", nil)
	req.Header.Set("Authorization", "Bearer dashboard-user-jwt")
	recorder := httptest.NewRecorder()

	mux.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusForbidden)
	}
	if calls != 0 {
		t.Fatalf("upstream calls = %d, want 0", calls)
	}
}

func TestRouterManagementProxyAllowlistMatchesDashboardSurfaces(t *testing.T) {
	tests := []struct {
		method string
		path   string
		want   bool
	}{
		{method: http.MethodGet, path: "/api/router/v1/models", want: true},
		{method: http.MethodGet, path: "/api/router/v1/router_replay", want: true},
		{method: http.MethodGet, path: "/api/router/v1/router_replay/replay-1", want: true},
		{method: http.MethodHead, path: "/api/router/v1/router_replay", want: false},
		{method: http.MethodPost, path: "/api/router/v1/router_replay", want: false},
		{method: http.MethodPost, path: "/api/router/v1/router/outcomes", want: true},
		{method: http.MethodGet, path: "/api/router/api/v1/response-cache/stats", want: true},
		{method: http.MethodPost, path: "/api/router/api/v1/response-cache/invalidate", want: true},
		{method: http.MethodPost, path: "/api/router/api/v1/context-compression/preview", want: true},
		{method: http.MethodDelete, path: "/api/router/api/v1/context-compression/stats", want: false},
		{method: http.MethodPost, path: "/api/router/config/router", want: false},
		{method: http.MethodPost, path: "/api/router/unknown", want: false},
	}
	for _, test := range tests {
		if got := routerManagementProxyRouteAllowed(test.method, test.path); got != test.want {
			t.Fatalf("routerManagementProxyRouteAllowed(%q, %q) = %v, want %v", test.method, test.path, got, test.want)
		}
	}
}

func TestRedactCredentialParams(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		raw  string
		want string
	}{
		{
			name: "authToken is redacted and other params survive",
			raw:  "http://localhost:8711/embedded/grafana/x?orgId=1&authToken=secret",
			want: "http://localhost:8711/embedded/grafana/x?authToken=%5BREDACTED%5D&orgId=1",
		},
		{
			name: "token is redacted",
			raw:  "http://localhost:8711/x?token=secret",
			want: "http://localhost:8711/x?token=%5BREDACTED%5D",
		},
		{
			name: "access_token is redacted",
			raw:  "http://localhost:8711/x?access_token=secret",
			want: "http://localhost:8711/x?access_token=%5BREDACTED%5D",
		},
		{
			name: "repeated authToken collapses to one redaction",
			raw:  "http://localhost:8711/x?authToken=a&authToken=b",
			want: "http://localhost:8711/x?authToken=%5BREDACTED%5D",
		},
		{
			name: "fragment is preserved",
			raw:  "http://localhost:8711/x?authToken=secret#gatewayUrl=http://localhost:8080",
			want: "http://localhost:8711/x?authToken=%5BREDACTED%5D#gatewayUrl=http://localhost:8080",
		},
		{
			name: "no credential is returned byte for byte",
			raw:  "http://localhost:8711/embedded/grafana/x?orgId=1&b=2",
			want: "http://localhost:8711/embedded/grafana/x?orgId=1&b=2",
		},
		{name: "empty", raw: "", want: ""},
		{name: "unparsable", raw: "::::", want: "[unparsable]"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if got := redactCredentialParams(tc.raw); got != tc.want {
				t.Fatalf("redactCredentialParams(%q) = %q, want %q", tc.raw, got, tc.want)
			}
		})
	}
}

func TestRedactCredentialParamsRemovesTheTokenFromTheLoggedReferer(t *testing.T) {
	t.Parallel()

	fakeJWT := "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature"
	logged := redactCredentialParams("http://localhost:8711/embedded/grafana/goto/x?orgId=1&authToken=" + fakeJWT)
	if strings.Contains(logged, fakeJWT) {
		t.Fatalf("the token survived redaction: %q", logged)
	}
}
