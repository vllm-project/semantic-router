package proxy

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
)

type capturedProxyRequest struct {
	path          string
	rawQuery      string
	authorization string
	openClawToken string
	referer       string
	cookies       []string
}

func TestReverseProxyDoesNotForwardDashboardCredentials(t *testing.T) {
	captured := make(chan capturedProxyRequest, 1)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured <- captureProxyRequest(r)
		w.Header().Add("Set-Cookie", "vsr_session=attacker; Path=/; HttpOnly")
		w.Header().Add("Set-Cookie", "grafana_session=g; Path=/embedded/test; HttpOnly")
		w.WriteHeader(http.StatusNoContent)
	}))
	defer upstream.Close()

	handler, err := NewReverseProxy(upstream.URL, "/embedded/test", false)
	if err != nil {
		t.Fatalf("NewReverseProxy: %v", err)
	}
	req := httptest.NewRequest(
		http.MethodGet,
		"http://dashboard.test/embedded/test/socket?theme=dark&auth%54oken=JWT&theme=light",
		nil,
	)
	req.Header.Set("Authorization", "Bearer dashboard-jwt")
	req.Header.Set("Referer", "http://dashboard.test/embedded/test/?theme=dark&authToken=JWT")
	req.Header.Add("Cookie", "vsr_session=first; grafana_session=g")
	req.Header.Add("Cookie", "theme=d; vsr_session=second")
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, req)

	got := <-captured
	assertSanitizedProxyRequest(t, got, "/socket", "theme=dark&theme=light", "grafana_session=g", "theme=d")
	if got.authorization != "" {
		t.Fatalf("dashboard Authorization leaked upstream: %q", got.authorization)
	}
	if got.referer != "" {
		t.Fatalf("dashboard Referer leaked upstream: %q", got.referer)
	}
	setCookies := recorder.Result().Header.Values("Set-Cookie")
	if len(setCookies) != 1 || !strings.HasPrefix(setCookies[0], "grafana_session=g;") {
		t.Fatalf("downstream Set-Cookie headers = %#v", setCookies)
	}
}

func TestReverseProxyOnlyForwardsAuthorizationOutsideDashboardBearerAuth(t *testing.T) {
	captured := make(chan string, 2)
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured <- r.Header.Get("Authorization")
		w.WriteHeader(http.StatusNoContent)
	}))
	defer upstream.Close()

	handler, err := NewReverseProxy(upstream.URL, "/api/router", true)
	if err != nil {
		t.Fatalf("NewReverseProxy: %v", err)
	}

	unauthenticated := httptest.NewRequest(http.MethodGet, "http://dashboard.test/api/router/health", nil)
	unauthenticated.Header.Set("Authorization", "Bearer upstream-api-key")
	handler.ServeHTTP(httptest.NewRecorder(), unauthenticated)
	if got := <-captured; got != "Bearer upstream-api-key" {
		t.Fatalf("auth-disabled upstream Authorization = %q", got)
	}

	authenticated := httptest.NewRequest(http.MethodGet, "http://dashboard.test/api/router/health", nil)
	authenticated.Header.Set("Authorization", "Bearer dashboard-jwt")
	authenticated = authenticated.WithContext(auth.WithAuthContext(authenticated.Context(), auth.AuthContext{
		CredentialSource: auth.CredentialSourceBearer,
	}))
	handler.ServeHTTP(httptest.NewRecorder(), authenticated)
	if got := <-captured; got != "" {
		t.Fatalf("dashboard bearer leaked upstream: %q", got)
	}
}

func TestReverseProxyOnlyAllowsSameOriginCredentialedCORS(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "https://attacker.example")
		w.Header().Set("Access-Control-Allow-Credentials", "true")
		w.WriteHeader(http.StatusNoContent)
	}))
	defer upstream.Close()
	handler, err := NewReverseProxy(upstream.URL, "/embedded/test", false)
	if err != nil {
		t.Fatalf("NewReverseProxy: %v", err)
	}

	for _, test := range []struct {
		name       string
		origin     string
		wantOrigin string
	}{
		{name: "same origin", origin: "https://dashboard.example.com", wantOrigin: "https://dashboard.example.com"},
		{name: "same-site sibling origin", origin: "https://evil.example.com", wantOrigin: ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "http://dashboard.example.com/embedded/test/", nil)
			req.Header.Set("Origin", test.origin)
			req.Header.Set("X-Forwarded-Proto", "https")
			recorder := httptest.NewRecorder()
			handler.ServeHTTP(recorder, req)
			if got := recorder.Header().Get("Access-Control-Allow-Origin"); got != test.wantOrigin {
				t.Fatalf("Access-Control-Allow-Origin = %q, want %q", got, test.wantOrigin)
			}
			wantCredentials := ""
			if test.wantOrigin != "" {
				wantCredentials = "true"
			}
			if got := recorder.Header().Get("Access-Control-Allow-Credentials"); got != wantCredentials {
				t.Fatalf("Access-Control-Allow-Credentials = %q, want %q", got, wantCredentials)
			}
		})
	}
}

func TestEmbeddedReverseProxyBlocksUpstreamServiceWorkers(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Service-Worker-Allowed", "/")
		w.Header().Set(
			"Content-Security-Policy",
			"default-src 'self'; worker-src 'self'; frame-ancestors 'none'",
		)
		w.WriteHeader(http.StatusNoContent)
	}))
	defer upstream.Close()

	handler, err := NewReverseProxy(upstream.URL, "/embedded/test", false)
	if err != nil {
		t.Fatalf("NewReverseProxy: %v", err)
	}
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(
		recorder,
		httptest.NewRequest(http.MethodGet, "http://dashboard.test/embedded/test/", nil),
	)

	if got := recorder.Header().Get("Service-Worker-Allowed"); got != "" {
		t.Fatalf("Service-Worker-Allowed = %q, want stripped", got)
	}
	csp := recorder.Header().Get("Content-Security-Policy")
	if !strings.Contains(csp, "worker-src 'none'") ||
		!strings.Contains(strings.ToLower(csp), "frame-ancestors 'self'") ||
		strings.Contains(csp, "worker-src 'self'") {
		t.Fatalf("Content-Security-Policy = %q, want same-origin framing and worker prohibition", csp)
	}
}

func TestEmbeddedReverseProxyPreservesMultipleCSPPolicies(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Add("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'; worker-src 'self'")
		w.Header().Add("Content-Security-Policy", "script-src 'none'; FRAME-ANCESTORS https://blocked.example")
		w.WriteHeader(http.StatusNoContent)
	}))
	defer upstream.Close()

	handler, err := NewReverseProxy(upstream.URL, "/embedded/test", false)
	if err != nil {
		t.Fatalf("NewReverseProxy: %v", err)
	}
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(
		recorder,
		httptest.NewRequest(http.MethodGet, "http://dashboard.test/embedded/test/", nil),
	)

	policies := recorder.Result().Header.Values("Content-Security-Policy")
	if len(policies) != 2 {
		t.Fatalf("Content-Security-Policy values = %#v, want two independent policies", policies)
	}
	if !strings.Contains(policies[0], "default-src 'self'") ||
		!strings.Contains(policies[1], "script-src 'none'") {
		t.Fatalf("Content-Security-Policy values lost upstream directives: %#v", policies)
	}
	for _, policy := range policies {
		if !strings.Contains(strings.ToLower(policy), "frame-ancestors 'self'") ||
			strings.Contains(strings.ToLower(policy), "frame-ancestors 'none'") ||
			strings.Contains(policy, "https://blocked.example") {
			t.Fatalf("Content-Security-Policy did not constrain framing to self: %q", policy)
		}
		if !strings.Contains(policy, "worker-src 'none'") || strings.Contains(policy, "worker-src 'self'") {
			t.Fatalf("Content-Security-Policy does not prohibit workers: %q", policy)
		}
	}
}

func TestEmbeddedReverseProxyTransformsCombinedCSPPolicyList(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set(
			"Content-Security-Policy",
			"default-src 'self'; frame-ancestors 'none'; worker-src 'self', script-src 'none'; frame-ancestors https://blocked.example",
		)
		w.WriteHeader(http.StatusNoContent)
	}))
	defer upstream.Close()

	handler, err := NewReverseProxy(upstream.URL, "/embedded/test", false)
	if err != nil {
		t.Fatalf("NewReverseProxy: %v", err)
	}
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(
		recorder,
		httptest.NewRequest(http.MethodGet, "http://dashboard.test/embedded/test/", nil),
	)

	policyList := recorder.Header().Get("Content-Security-Policy")
	policies := strings.Split(policyList, ",")
	if len(policies) != 2 {
		t.Fatalf("Content-Security-Policy = %q, want two combined policies", policyList)
	}
	for _, policy := range policies {
		lower := strings.ToLower(policy)
		if !strings.Contains(lower, "frame-ancestors 'self'") ||
			!strings.Contains(lower, "worker-src 'none'") ||
			strings.Contains(lower, "frame-ancestors 'none'") ||
			strings.Contains(lower, "https://blocked.example") ||
			strings.Contains(lower, "worker-src 'self'") {
			t.Fatalf("combined Content-Security-Policy was not transformed independently: %q", policy)
		}
	}
}

func TestNonEmbeddedReverseProxyPreservesUpstreamWorkerPolicy(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Service-Worker-Allowed", "/")
		w.Header().Set("Content-Security-Policy", "default-src 'self'; worker-src 'self'")
		w.WriteHeader(http.StatusNoContent)
	}))
	defer upstream.Close()

	handler, err := NewReverseProxy(upstream.URL, "/api/router", false)
	if err != nil {
		t.Fatalf("NewReverseProxy: %v", err)
	}
	recorder := httptest.NewRecorder()
	handler.ServeHTTP(
		recorder,
		httptest.NewRequest(http.MethodGet, "http://dashboard.test/api/router/status", nil),
	)

	if got := recorder.Header().Get("Service-Worker-Allowed"); got != "" {
		t.Fatalf("Service-Worker-Allowed = %q, want stripped", got)
	}
	csp := recorder.Header().Get("Content-Security-Policy")
	if !strings.Contains(csp, "worker-src 'self'") || strings.Contains(csp, "worker-src 'none'") {
		t.Fatalf("Content-Security-Policy = %q, want non-embedded worker policy preserved", csp)
	}
}

func captureProxyRequest(r *http.Request) capturedProxyRequest {
	return capturedProxyRequest{
		path:          r.URL.Path,
		rawQuery:      r.URL.RawQuery,
		authorization: r.Header.Get("Authorization"),
		openClawToken: r.Header.Get("X-OpenClaw-Token"),
		referer:       r.Header.Get("Referer"),
		cookies:       append([]string(nil), r.Header.Values("Cookie")...),
	}
}

func assertSanitizedProxyRequest(
	t *testing.T,
	got capturedProxyRequest,
	wantPath string,
	wantRawQuery string,
	wantCookies ...string,
) {
	t.Helper()
	if got.path != wantPath || got.rawQuery != wantRawQuery {
		t.Fatalf("upstream target = %s?%s, want %s?%s", got.path, got.rawQuery, wantPath, wantRawQuery)
	}
	joined := strings.Join(got.cookies, "; ")
	if strings.Contains(joined, dashboardSessionCookieName+"=") {
		t.Fatalf("dashboard session cookie leaked upstream: %#v", got.cookies)
	}
	for _, want := range wantCookies {
		if !strings.Contains(joined, want) {
			t.Fatalf("upstream Cookie headers %#v do not contain %q", got.cookies, want)
		}
	}
}
