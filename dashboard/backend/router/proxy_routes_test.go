package router

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	dashboardauth "github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/proxy"
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

func TestSmartAPIRouterOnlyForwardsRecognizedGrafanaAPIPaths(t *testing.T) {
	t.Parallel()
	var calls int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		w.WriteHeader(http.StatusNoContent)
	}))
	defer upstream.Close()
	proxy, err := proxy.NewReverseProxy(upstream.URL, "", false)
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	registerSmartAPIRouter(mux, dashboardProxySet{grafanaStatic: proxy})
	for _, path := range []string{"/api/application/users", "/api/inference/models", "/api/unknown"} {
		recorder := httptest.NewRecorder()
		mux.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, path, nil))
		if recorder.Code != http.StatusNotFound {
			t.Fatalf("%s status=%d, want 404", path, recorder.Code)
		}
	}
	if calls != 0 {
		t.Fatalf("unrecognized API calls reached fallback upstream: %d", calls)
	}
}

func TestSmartAPIRouterForwardsRecognizedGrafanaAPIPath(t *testing.T) {
	t.Parallel()
	var receivedPath string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedPath = r.URL.Path
		w.WriteHeader(http.StatusNoContent)
	}))
	defer upstream.Close()
	upstreamProxy, err := proxy.NewReverseProxy(upstream.URL, "", false)
	if err != nil {
		t.Fatal(err)
	}
	mux := http.NewServeMux()
	registerSmartAPIRouter(mux, dashboardProxySet{grafanaStatic: upstreamProxy})
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/api/dashboards/uid/router", nil))
	if recorder.Code != http.StatusNoContent || receivedPath != "/api/dashboards/uid/router" {
		t.Fatalf("status=%d path=%q", recorder.Code, receivedPath)
	}
}

func TestRegisterFleetSimRoutesReturnsBadGatewayWhenDisabled(t *testing.T) {
	t.Parallel()
	mux := http.NewServeMux()
	registerFleetSimRoutes(mux, &config.Config{})
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/api/fleet-sim/api/workloads", nil))
	if recorder.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusBadGateway)
	}
}

func TestRegisterFleetSimRoutesProxiesSimulatorPaths(t *testing.T) {
	t.Parallel()
	var proxiedPath, forwardedPrefix string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		proxiedPath, forwardedPrefix = r.URL.Path, r.Header.Get("X-Forwarded-Prefix")
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer server.Close()
	mux := http.NewServeMux()
	registerFleetSimRoutes(mux, &config.Config{FleetSimURL: server.URL})
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/api/fleet-sim/api/workloads", nil))
	if recorder.Code != http.StatusOK || proxiedPath != "/api/workloads" || forwardedPrefix != "/api/fleet-sim" {
		t.Fatalf("status=%d path=%q prefix=%q", recorder.Code, proxiedPath, forwardedPrefix)
	}
}

type testManagementSessions struct{ token string }

func (provider testManagementSessions) ManagementAccessToken(context.Context, dashboardauth.AuthContext) (string, error) {
	return provider.token, nil
}

func TestRouterManagementProxyUsesPrincipalSessionAndStripsBrowserCredentials(t *testing.T) {
	var path, authorization, cookie, proxyAuthorization, queryCredential string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path, authorization = r.URL.RequestURI(), r.Header.Get("Authorization")
		cookie, proxyAuthorization = r.Header.Get("Cookie"), r.Header.Get("Proxy-Authorization")
		queryCredential = r.URL.Query().Get("authToken")
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()
	mux := http.NewServeMux()
	registerRouterManagementProxy(mux, &config.Config{RouterAPIURL: server.URL}, testManagementSessions{token: "principal-token"})
	request := httptest.NewRequest(http.MethodGet, "/api/router/management/v1/users?authToken=browser-query&cursor=next", nil)
	request.Header.Set("Authorization", "Bearer dashboard-jwt")
	request.Header.Set("Proxy-Authorization", "Bearer proxy-jwt")
	request.Header.Set("Cookie", "vsr_session=dashboard-jwt")
	request = request.WithContext(dashboardauth.WithAuthContext(request.Context(), dashboardauth.AuthContext{
		UserID: "user-1", SessionID: "dashboard-session-1",
	}))
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, request)
	if recorder.Code != http.StatusNoContent {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusNoContent)
	}
	if path != "/management/v1/users?cursor=next" || authorization != "Bearer principal-token" {
		t.Fatalf("path=%q authorization=%q", path, authorization)
	}
	if cookie != "" || proxyAuthorization != "" || queryCredential != "" {
		t.Fatalf("browser credential leaked: cookie=%q proxy=%q query=%q", cookie, proxyAuthorization, queryCredential)
	}
}

func TestRouterManagementProxyFailsClosedWithoutPrincipalExchange(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()
	mux := http.NewServeMux()
	registerRouterManagementProxy(mux, &config.Config{RouterAPIURL: server.URL}, nil)
	recorder := httptest.NewRecorder()
	mux.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/api/router/management/v1/me", nil))
	if recorder.Code != http.StatusServiceUnavailable || calls != 0 {
		t.Fatalf("status=%d upstream calls=%d", recorder.Code, calls)
	}
}

func TestDashboardDoesNotProxyPublicInferenceAPI(t *testing.T) {
	var calls int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()
	mux := http.NewServeMux()
	registerRouterManagementProxy(mux, &config.Config{RouterAPIURL: server.URL}, testManagementSessions{token: "principal-token"})
	for _, path := range []string{"/api/router/v1/models", "/api/router/v1/chat/completions", "/api/playground/v1/models"} {
		recorder := httptest.NewRecorder()
		mux.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, path, nil))
		if recorder.Code != http.StatusNotFound {
			t.Fatalf("%s status=%d, want 404", path, recorder.Code)
		}
	}
	if calls != 0 {
		t.Fatalf("public inference upstream calls=%d, want 0", calls)
	}
}
