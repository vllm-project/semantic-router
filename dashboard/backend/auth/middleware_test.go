package auth

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRequiresAuthentication(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		path     string
		expected bool
	}{
		{path: "/", expected: false},
		{path: "/dashboard", expected: false},
		{path: "/login", expected: false},
		{path: "/api/auth/login", expected: false},
		{path: "/api/auth/logout", expected: false},
		{path: "/api/auth/bootstrap/can-register", expected: false},
		{path: "/api/setup/state", expected: false},
		{path: "/api/auth/me", expected: true},
		{path: "/api/status", expected: true},
		{path: "/embedded/grafana/", expected: true},
		{path: "/embedded/wizmap/", expected: true},
		{path: "/embedded/wizmap/assets/index.js", expected: false},
	}

	for _, tc := range testCases {
		t.Run(tc.path, func(t *testing.T) {
			t.Parallel()
			if actual := requiresAuthentication(tc.path); actual != tc.expected {
				t.Fatalf("requiresAuthentication(%q) = %v, want %v", tc.path, actual, tc.expected)
			}
		})
	}
}

func TestExtractAccessToken(t *testing.T) {
	t.Parallel()

	t.Run("prefers bearer header", func(t *testing.T) {
		t.Parallel()
		req := httptest.NewRequest(http.MethodGet, "/api/status?authToken=query-token", nil)
		req.Header.Set("Authorization", "Bearer header-token")

		if token := extractAccessToken(req); token != "header-token" {
			t.Fatalf("extractAccessToken() = %q, want header-token", token)
		}
	})

	t.Run("does not fall back to query token", func(t *testing.T) {
		t.Parallel()
		req := httptest.NewRequest(http.MethodGet, "/embedded/grafana/?authToken=query-token", nil)

		if token := extractAccessToken(req); token != "" {
			t.Fatalf("extractAccessToken() = %q, want empty", token)
		}
	})

	t.Run("falls back to cookie token", func(t *testing.T) {
		t.Parallel()
		req := httptest.NewRequest(http.MethodGet, "/embedded/grafana/?authToken=query-token", nil)
		req.AddCookie(&http.Cookie{Name: authSessionCookieName, Value: "cookie-token"})

		if token := extractAccessToken(req); token != "cookie-token" {
			t.Fatalf("extractAccessToken() = %q, want cookie-token", token)
		}
	})
}

func TestRequiredPermission(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		method   string
		path     string
		expected string
	}{
		{method: http.MethodGet, path: "/api/admin/users", expected: PermUsersView},
		{method: http.MethodPatch, path: "/api/admin/users/user-1", expected: PermUsersManage},
		{method: http.MethodGet, path: "/api/status", expected: PermLogsRead},
		{method: http.MethodGet, path: "/embedded/grafana/", expected: PermLogsRead},
		{method: http.MethodGet, path: "/embedded/wizmap/", expected: PermConfigRead},
		{method: http.MethodPost, path: "/api/setup/activate", expected: PermConfigWrite},
		{method: http.MethodPost, path: "/api/setup/import-remote", expected: PermConfigWrite},
		{method: http.MethodGet, path: "/api/mcp/servers", expected: PermMcpRead},
		{method: http.MethodPost, path: "/api/mcp/servers/server-1/connect", expected: PermMcpManage},
		{method: http.MethodGet, path: "/api/openclaw/teams", expected: PermOpenClawRead},
		{method: http.MethodPost, path: "/api/openclaw/teams", expected: PermOpenClaw},
		{method: http.MethodPost, path: "/api/openclaw/rooms/room-1/messages", expected: PermOpenClawRead},
		{method: http.MethodPost, path: "/api/router/v1/chat/completions", expected: PermConfigRead},
	}

	for _, tc := range testCases {
		t.Run(tc.path, func(t *testing.T) {
			t.Parallel()
			if actual := RequiredPermission(tc.method, tc.path); actual != tc.expected {
				t.Fatalf("RequiredPermission(%q, %q) = %q, want %q", tc.method, tc.path, actual, tc.expected)
			}
		})
	}
}
