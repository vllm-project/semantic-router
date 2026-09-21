package auth

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestServiceUnavailableGuard(t *testing.T) {
	t.Parallel()

	routes := NewPolicyMux()
	next := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	routes.HandleFunc(ProtectedRoute("/api/router/config/all", PermConfigRead, SensitivitySecret, ResourceOwnerConfig, http.MethodGet), next)
	routes.HandleFunc(ProtectedRoute("/api/admin/users", PermUsersView, SensitivitySensitive, ResourceOwnerAuth, http.MethodGet), next)
	routes.HandleFunc(ProtectedRoute("/embedded/grafana/", PermLogsRead, SensitivitySensitive, ResourceOwnerObservability, http.MethodGet), next)
	routes.HandleFunc(PublicRoute("/api/auth/login", http.MethodPost), next)
	routes.HandleFunc(PublicRoute("/api/setup/state", http.MethodGet), next)
	routes.HandleFunc(PublicRoute("/api/status", http.MethodGet), next)
	routes.HandleFallback("/", next)

	testCases := []struct {
		name     string
		method   string
		path     string
		wantCode int
	}{
		{name: "protected api denied", method: http.MethodGet, path: "/api/router/config/all", wantCode: http.StatusServiceUnavailable},
		{name: "admin denied", method: http.MethodGet, path: "/api/admin/users", wantCode: http.StatusServiceUnavailable},
		{name: "embedded denied", method: http.MethodGet, path: "/embedded/grafana/", wantCode: http.StatusServiceUnavailable},
		{name: "unregistered api denied", method: http.MethodGet, path: "/api/unregistered", wantCode: http.StatusServiceUnavailable},
		{name: "undeclared method rejected", method: http.MethodDelete, path: "/api/admin/users", wantCode: http.StatusMethodNotAllowed},
		{name: "login public", method: http.MethodPost, path: "/api/auth/login", wantCode: http.StatusOK},
		{name: "setup state public", method: http.MethodGet, path: "/api/setup/state", wantCode: http.StatusOK},
		{name: "system status public", method: http.MethodGet, path: "/api/status", wantCode: http.StatusOK},
		{name: "static frontend public", method: http.MethodGet, path: "/dashboard", wantCode: http.StatusOK},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			rec := httptest.NewRecorder()
			ServiceUnavailableGuard(routes)(routes).ServeHTTP(rec, httptest.NewRequest(tc.method, tc.path, nil))
			if rec.Code != tc.wantCode {
				t.Fatalf("status = %d, want %d", rec.Code, tc.wantCode)
			}
		})
	}
}

func TestExtractAccessToken(t *testing.T) {
	t.Parallel()

	t.Run("prefers bearer header", func(t *testing.T) {
		t.Parallel()
		req := httptest.NewRequest(http.MethodGet, "/api/status?authToken=query-token", nil)
		req.Header.Set("Authorization", "Bearer  header-token ")

		if token := extractAccessToken(req); token != "header-token" {
			t.Fatalf("extractAccessToken() = %q, want header-token", token)
		}
	})

	// Inverted for #2465: this asserted the query token was returned.
	t.Run("ignores a well-formed query token", func(t *testing.T) {
		t.Parallel()
		req := httptest.NewRequest(http.MethodGet, "/embedded/grafana/?authToken=query-token", nil)

		if token := extractAccessToken(req); token != "" {
			t.Fatalf("extractAccessToken() = %q, want empty", token)
		}
	})

	t.Run("prefers the cookie token and ignores the query token", func(t *testing.T) {
		t.Parallel()
		req := httptest.NewRequest(http.MethodGet, "/embedded/grafana/?authToken=query-token", nil)
		req.AddCookie(&http.Cookie{Name: authSessionCookieName, Value: "cookie-token"})

		if token := extractAccessToken(req); token != "cookie-token" {
			t.Fatalf("extractAccessToken() = %q, want cookie-token", token)
		}
	})

	t.Run("skips malformed bearer and uses cookie token", func(t *testing.T) {
		t.Parallel()
		req := httptest.NewRequest(http.MethodGet, "/api/status?authToken=query-token", nil)
		req.Header.Set("Authorization", "Bearer invalid token")
		req.AddCookie(&http.Cookie{Name: authSessionCookieName, Value: "cookie-token"})

		if token := extractAccessToken(req); token != "cookie-token" {
			t.Fatalf("extractAccessToken() = %q, want cookie-token", token)
		}
	})

	// "Empty" is not enough on its own: assert the source too, or a reinstated query
	// branch could pass this silently.
	t.Run("reports no source for any query token shape", func(t *testing.T) {
		t.Parallel()

		for _, raw := range []string{"query-token", "invalid%20token", ""} {
			req := httptest.NewRequest(http.MethodGet, "/embedded/grafana/?authToken="+raw, nil)

			token, source := extractAccessTokenWithSource(req)
			if token != "" || source != tokenSourceNone {
				t.Fatalf("authToken=%q: got (%q, %v), want (\"\", tokenSourceNone)", raw, token, source)
			}
		}
	})
}

func TestNormalizeAccessToken(t *testing.T) {
	t.Parallel()

	if token := normalizeAccessToken("  header-token_123.abc-def  "); token != "header-token_123.abc-def" {
		t.Fatalf("normalizeAccessToken() = %q, want trimmed token", token)
	}

	testCases := []struct {
		name string
		raw  string
	}{
		{name: "empty", raw: ""},
		{name: "space", raw: "invalid token"},
		{name: "tab", raw: "invalid\ttoken"},
		{name: "newline", raw: "invalid\ntoken"},
		{name: "semicolon", raw: "invalid;token"},
		{name: "oversized", raw: strings.Repeat("a", maxAccessTokenBytes+1)},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			if token := normalizeAccessToken(tc.raw); token != "" {
				t.Fatalf("normalizeAccessToken(%q) = %q, want empty", tc.raw, token)
			}
		})
	}
}
