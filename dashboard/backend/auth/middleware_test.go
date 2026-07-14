package auth

import (
	"net/http"
	"net/http/httptest"
	"strings"
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
		{path: "/api/auth/login/", expected: false},
		{path: "/api/auth/login-anything", expected: true},
		{path: "/api/auth/login/anything", expected: true},
		{path: "/api/auth/logout", expected: false},
		{path: "/api/auth/logout-anything", expected: true},
		{path: "/api/auth/logout/anything", expected: true},
		{path: "/api/auth/bootstrap/can-register", expected: false},
		{path: "/api/auth/bootstrap/can-register/", expected: false},
		{path: "/api/auth/bootstrap/register", expected: false},
		{path: "/api/auth/bootstrap/register/", expected: false},
		{path: "/api/auth/bootstrap/unknown", expected: true},
		{path: "/api/setup/state", expected: false},
		{path: "/api/setup/state-anything", expected: true},
		{path: "/api/auth/me", expected: true},
		{path: "/api/auth/password", expected: true},
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

func TestServiceUnavailableGuard(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		path     string
		wantCode int
		wantNext bool
	}{
		{name: "protected api denied", path: "/api/router/config", wantCode: http.StatusServiceUnavailable, wantNext: false},
		{name: "admin denied", path: "/api/admin/users", wantCode: http.StatusServiceUnavailable, wantNext: false},
		{name: "embedded denied", path: "/embedded/grafana/", wantCode: http.StatusServiceUnavailable, wantNext: false},
		{name: "login public", path: "/api/auth/login", wantCode: http.StatusOK, wantNext: true},
		{name: "login suffix denied", path: "/api/auth/login-anything", wantCode: http.StatusServiceUnavailable, wantNext: false},
		{name: "bootstrap trailing slash public", path: "/api/auth/bootstrap/can-register/", wantCode: http.StatusOK, wantNext: true},
		{name: "bootstrap suffix denied", path: "/api/auth/bootstrap/unknown", wantCode: http.StatusServiceUnavailable, wantNext: false},
		{name: "setup state public", path: "/api/setup/state", wantCode: http.StatusOK, wantNext: true},
		{name: "static frontend public", path: "/dashboard", wantCode: http.StatusOK, wantNext: true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			nextCalled := false
			next := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				nextCalled = true
				w.WriteHeader(http.StatusOK)
			})
			handler := ServiceUnavailableGuard()(next)

			req := httptest.NewRequest(http.MethodGet, tc.path, nil)
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, req)

			if rec.Code != tc.wantCode {
				t.Fatalf("status = %d, want %d", rec.Code, tc.wantCode)
			}
			if nextCalled != tc.wantNext {
				t.Fatalf("next handler called = %v, want %v", nextCalled, tc.wantNext)
			}
			wantCacheControl := ""
			if requiresAuthentication(tc.path) {
				wantCacheControl = "no-store"
			}
			if got := rec.Header().Get("Cache-Control"); got != wantCacheControl {
				t.Fatalf("Cache-Control = %q, want %q", got, wantCacheControl)
			}
		})
	}
}

func TestRuntimeConfigApplyRequiresDeployAuthority(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "config-write-only@example.com", RoleRead, "active")
	if _, err := svc.store.db.Exec(
		`INSERT INTO user_permissions(user_id, permission_key, allowed) VALUES(?,?,1)`,
		user.ID,
		PermConfigWrite,
	); err != nil {
		t.Fatalf("grant config.write: %v", err)
	}
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issue token: %v", err)
	}

	paths := []string{
		"/api/setup/activate",
		"/api/router/config/update",
		"/api/router/config/global/update",
		"/api/router/config/global/raw/update",
		"/api/router/config/defaults/update",
	}
	for _, path := range paths {
		t.Run(path, func(t *testing.T) {
			called := false
			handler := AuthenticateRequest(svc)(http.HandlerFunc(func(
				w http.ResponseWriter,
				_ *http.Request,
			) {
				called = true
				w.WriteHeader(http.StatusNoContent)
			}))
			req := httptest.NewRequest(http.MethodPost, path, nil)
			req.Header.Set("Authorization", "Bearer "+token)
			rec := httptest.NewRecorder()

			handler.ServeHTTP(rec, req)

			if rec.Code != http.StatusForbidden {
				t.Fatalf("status = %d, want %d", rec.Code, http.StatusForbidden)
			}
			if called {
				t.Fatal("runtime apply handler ran without config.deploy")
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

	t.Run("rejects query token", func(t *testing.T) {
		t.Parallel()
		req := httptest.NewRequest(http.MethodGet, "/embedded/grafana/?authToken=query-token", nil)

		if token := extractAccessToken(req); token != "" {
			t.Fatalf("extractAccessToken() = %q, want empty", token)
		}
	})

	t.Run("uses cookie while ignoring query token", func(t *testing.T) {
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

	t.Run("rejects malformed query token", func(t *testing.T) {
		t.Parallel()
		req := httptest.NewRequest(http.MethodGet, "/embedded/grafana/?authToken=invalid%20token", nil)

		if token := extractAccessToken(req); token != "" {
			t.Fatalf("extractAccessToken() = %q, want empty", token)
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

func TestRequiredPermission(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		method   string
		path     string
		expected string
	}{
		{method: http.MethodGet, path: "/api/admin/users", expected: PermUsersView},
		{method: http.MethodPost, path: "/api/auth/password", expected: ""},
		{method: http.MethodPatch, path: "/api/admin/users/user-1", expected: PermUsersManage},
		{method: http.MethodGet, path: "/api/admin/audit-logs", expected: PermUsersManage},
		{method: http.MethodGet, path: "/api/status", expected: PermLogsRead},
		{method: http.MethodGet, path: "/embedded/grafana/", expected: PermLogsRead},
		{method: http.MethodGet, path: "/embedded/wizmap/", expected: PermConfigRead},
		{method: http.MethodPost, path: "/api/setup/activate", expected: PermConfigDeploy},
		{method: http.MethodPost, path: "/api/setup/import-remote", expected: PermConfigWrite},
		{method: http.MethodGet, path: "/api/mcp/servers", expected: PermMcpRead},
		{method: http.MethodPost, path: "/api/mcp/servers", expected: PermMcpManage},
		{method: http.MethodDelete, path: "/api/mcp/servers/server-1/status", expected: PermMcpManage},
		{method: http.MethodPost, path: "/api/mcp/servers/server-1/connect", expected: PermMcpManage},
		{method: http.MethodPost, path: "/api/router/config/deploy", expected: PermConfigDeploy},
		{method: http.MethodPost, path: "/api/router/config/deploy/preview", expected: PermConfigDeploy},
		{method: http.MethodPost, path: "/api/router/config/update", expected: PermConfigDeploy},
		{method: http.MethodPost, path: "/api/router/config/global/update", expected: PermConfigDeploy},
		{method: http.MethodPost, path: "/api/router/config/global/raw/update", expected: PermConfigDeploy},
		{method: http.MethodPost, path: "/api/router/config/defaults/update", expected: PermConfigDeploy},
		{method: http.MethodGet, path: "/api/router/config/deployments", expected: PermConfigRead},
		{method: http.MethodPost, path: "/api/evaluation/tasks", expected: PermEvalWrite},
		{method: http.MethodPost, path: "/api/evaluation/run", expected: PermEvalRun},
		{method: http.MethodPost, path: "/api/evaluation/cancel/task-1", expected: PermEvalRun},
		{method: http.MethodGet, path: "/api/fleet-sim/api/workloads", expected: PermConfigRead},
		{method: http.MethodPost, path: "/api/fleet-sim/api/jobs", expected: PermConfigWrite},
		{method: http.MethodGet, path: "/api/openclaw/teams", expected: PermOpenClawRead},
		{method: http.MethodPost, path: "/api/openclaw/teams", expected: PermOpenClaw},
		{method: http.MethodPost, path: "/api/openclaw/mcp", expected: PermOpenClaw},
		{method: http.MethodPost, path: "/api/openclaw/rooms/room-1/messages", expected: PermOpenClawUse},
		{method: http.MethodGet, path: "/api/openclaw/rooms/room-1/messages", expected: PermOpenClawRead},
		{method: http.MethodGet, path: "/api/openclaw/token", expected: PermOpenClaw},
		{method: http.MethodGet, path: "/embedded/openclaw/worker-1/", expected: PermOpenClaw},
		{method: http.MethodPost, path: "/api/router/v1/chat/completions", expected: PermConfigRead},
		{method: http.MethodGet, path: "/api/security/policy", expected: PermConfigRead},
		{method: http.MethodPut, path: "/api/security/policy", expected: PermSecurityManage},
		{method: http.MethodPost, path: "/api/security/policy/preview", expected: PermSecurityManage},
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
