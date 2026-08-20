package router

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

func TestSmartAPIRouterReturnsNotFoundForUnknownAPIPath(t *testing.T) {
	t.Parallel()
	mux := http.NewServeMux()
	registerSmartAPIRouter(mux, dashboardProxySet{})

	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/api/security-policy", nil))

	if recorder.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusNotFound)
	}
	if recorder.Header().Get("Content-Type") != "application/json" {
		t.Fatalf("content type = %q, want application/json", recorder.Header().Get("Content-Type"))
	}
}

func TestRegisterFleetSimRoutesReturnsBadGatewayWhenDisabled(t *testing.T) {
	t.Parallel()

	mux := http.NewServeMux()
	registerFleetSimRoutes(mux, &config.Config{})

	req := httptest.NewRequest(http.MethodGet, "/api/fleet-sim/api/workloads", nil)
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusBadGateway)
	}
}

func TestRegisterFleetSimRoutesProxiesSimulatorPaths(t *testing.T) {
	t.Parallel()

	var proxiedPath string
	var forwardedPrefix string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxiedPath = r.URL.Path
		forwardedPrefix = r.Header.Get("X-Forwarded-Prefix")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer server.Close()

	mux := http.NewServeMux()
	registerFleetSimRoutes(mux, &config.Config{FleetSimURL: server.URL})

	req := httptest.NewRequest(http.MethodGet, "/api/fleet-sim/api/workloads", nil)
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
	if proxiedPath != "/api/workloads" {
		t.Fatalf("proxied path = %q, want %q", proxiedPath, "/api/workloads")
	}
	if forwardedPrefix != "/api/fleet-sim" {
		t.Fatalf("forwarded prefix = %q, want %q", forwardedPrefix, "/api/fleet-sim")
	}
}

type routerProxyCredentialProvider struct {
	token string
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
