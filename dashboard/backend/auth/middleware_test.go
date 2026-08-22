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
		{name: "invitation acceptance public", path: "/api/auth/invitations/info", wantCode: http.StatusOK, wantNext: true},
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

	t.Run("falls back to query token", func(t *testing.T) {
		t.Parallel()
		req := httptest.NewRequest(http.MethodGet, "/embedded/grafana/?authToken=query-token", nil)

		if token := extractAccessToken(req); token != "query-token" {
			t.Fatalf("extractAccessToken() = %q, want query-token", token)
		}
	})

	t.Run("falls back to cookie token before query token", func(t *testing.T) {
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
		{method: http.MethodPatch, path: "/api/admin/users/user-1", expected: PermUsersManage},
		{method: http.MethodGet, path: "/api/admin/audit-logs", expected: PermUsersManage},
		{method: http.MethodGet, path: "/api/status", expected: PermStatusRead},
		{method: http.MethodGet, path: "/embedded/grafana/", expected: PermLogsRead},
		{method: http.MethodGet, path: "/embedded/wizmap/", expected: PermConfigRead},
		{method: http.MethodPost, path: "/api/setup/activate", expected: PermConfigWrite},
		{method: http.MethodPost, path: "/api/setup/import-remote", expected: PermConfigWrite},
		{method: http.MethodGet, path: "/api/models/catalog", expected: PermConfigRead},
		{method: http.MethodPost, path: "/api/models/verify", expected: PermStatusRead},
		{method: http.MethodGet, path: "/api/models/verify", expected: PermStatusRead},
		{method: http.MethodGet, path: "/api/mcp/servers", expected: PermMcpRead},
		{method: http.MethodPost, path: "/api/mcp/servers", expected: PermMcpManage},
		{method: http.MethodDelete, path: "/api/mcp/servers/server-1/status", expected: PermMcpManage},
		{method: http.MethodPost, path: "/api/mcp/servers/server-1/connect", expected: PermMcpManage},
		{method: http.MethodPost, path: "/api/router/config/deploy", expected: PermConfigDeploy},
		{method: http.MethodPost, path: "/api/router/config/deploy/preview", expected: PermConfigDeploy},
		{method: http.MethodGet, path: "/api/router/config/deployments", expected: PermConfigRead},
		{method: http.MethodGet, path: "/api/router/api/v1/response-cache/stats", expected: PermConfigRead},
		{method: http.MethodPost, path: "/api/router/api/v1/response-cache/invalidate", expected: PermConfigWrite},
		{method: http.MethodPost, path: "/api/router/api/v1/context-compression/preview", expected: PermConfigRead},
		{method: http.MethodPost, path: "/api/router/api/v1/context-compression/recovery/invalidate", expected: PermConfigWrite},
		{method: http.MethodPost, path: "/api/evaluation/tasks", expected: PermEvalWrite},
		{method: http.MethodPost, path: "/api/evaluation/run", expected: PermEvalRun},
		{method: http.MethodPost, path: "/api/evaluation/cancel/task-1", expected: PermEvalRun},
		{method: http.MethodGet, path: "/api/fleet-sim/api/workloads", expected: PermConfigRead},
		{method: http.MethodPost, path: "/api/fleet-sim/api/jobs", expected: PermConfigWrite},
		{method: http.MethodGet, path: "/api/openclaw/teams", expected: PermOpenClawRead},
		{method: http.MethodPost, path: "/api/openclaw/teams", expected: PermOpenClaw},
		{method: http.MethodPost, path: "/api/openclaw/rooms/room-1/messages", expected: PermOpenClawRead},
		{method: http.MethodPost, path: "/api/router/v1/chat/completions", expected: PermConfigRead},
		{method: http.MethodPost, path: "/api/router/v1/router/outcomes", expected: PermFeedbackSubmit},
		{method: http.MethodGet, path: "/api/router/v1/router_replay", expected: PermReplayRead},
		{method: http.MethodGet, path: "/api/router/v1/router_replay/record-1", expected: PermReplayRead},
		{method: http.MethodGet, path: "/api/recipe", expected: PermConfigRead},
		{method: http.MethodGet, path: "/api/recipe-drafts", expected: PermConfigRead},
		{method: http.MethodPut, path: "/api/recipe-drafts/speed", expected: PermConfigWrite},
		{method: http.MethodGet, path: "/api/recipe/probes", expected: PermConfigRead},
		{method: http.MethodGet, path: "/api/recipe/packages", expected: PermConfigRead},
		{method: http.MethodGet, path: "/api/recipe/packages/", expected: PermConfigRead},
		{method: http.MethodGet, path: "/api/recipe/packages/anything", expected: PermConfigRead},
		{method: http.MethodPost, path: "/api/recipe/import", expected: PermConfigWrite},
		{method: http.MethodPost, path: "/api/recipe/import/", expected: PermConfigWrite},
		{method: http.MethodPost, path: "/api/recipe/import/anything", expected: PermConfigWrite},
		{method: http.MethodPost, path: "/api/recipe/activate", expected: PermConfigDeploy},
		{method: http.MethodPost, path: "/api/recipe/activate/preview", expected: PermConfigDeploy},
		{method: http.MethodPost, path: "/api/recipe/activate/", expected: PermConfigDeploy},
		{method: http.MethodPost, path: "/api/recipe/activate/anything", expected: PermConfigDeploy},
		{method: http.MethodPost, path: "/api/recipe/deactivate", expected: PermConfigDeploy},
		{method: http.MethodPost, path: "/api/recipe/deactivate/preview", expected: PermConfigDeploy},
		{method: http.MethodPost, path: "/api/recipe/deactivate/", expected: PermConfigDeploy},
		{method: http.MethodPost, path: "/api/recipe/deactivate/anything", expected: PermConfigDeploy},
		{method: http.MethodPost, path: "/api/recipe/probes/lane/variant/run-plan", expected: PermConfigRead},
		{method: http.MethodPost, path: "/api/recipe/probes/lane/variant/validate", expected: PermTopologyRead},
		{method: http.MethodPost, path: "/api/recipe/probes/lane/variant/validate/", expected: PermTopologyRead},
		{method: http.MethodPost, path: "/api/recipe/probes/lane/variant/validate///", expected: PermTopologyRead},
		{method: http.MethodGet, path: "/api/v1/access-control/api-keys", expected: PermAccessRead},
		{method: http.MethodPost, path: "/api/v1/access-control/api-keys", expected: PermAccessManage},
		{method: http.MethodGet, path: "/api/v1/access-control/usage", expected: PermAccessRead},
		{method: http.MethodGet, path: "/api/v1/access-control/request-logs", expected: PermAccessRead},
		{method: http.MethodGet, path: "/api/v1/access-control/audit-logs", expected: PermAccessRead},
		{method: http.MethodGet, path: "/api/v1/access-control/self/api-keys", expected: PermAccessSelf},
		{method: http.MethodPost, path: "/api/v1/access-control/self/api-keys", expected: PermAccessSelf},
		{method: http.MethodGet, path: "/api/playground/v1/models", expected: PermAccessSelf},
		{method: http.MethodPost, path: "/api/playground/v1/chat/completions", expected: PermAccessSelf},
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

func TestAuthenticateRequestRequiresFeedbackPermissionForRouterOutcomes(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	reader := newTestUser(t, svc, "outcome-reader@example.com", RoleRead, "active")
	writer := newTestUser(t, svc, "outcome-writer@example.com", RoleWrite, "active")
	nextCalled := false
	handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		nextCalled = true
		w.WriteHeader(http.StatusNoContent)
	}))

	readerRecorder := httptest.NewRecorder()
	handler.ServeHTTP(
		readerRecorder,
		newAuthenticatedRequest(t, svc, reader, http.MethodPost, "/api/router/v1/router/outcomes", `{}`),
	)
	if readerRecorder.Code != http.StatusForbidden || nextCalled {
		t.Fatalf("read role status = %d, next called = %v", readerRecorder.Code, nextCalled)
	}

	writerRecorder := httptest.NewRecorder()
	handler.ServeHTTP(
		writerRecorder,
		newAuthenticatedRequest(t, svc, writer, http.MethodPost, "/api/router/v1/router/outcomes", `{}`),
	)
	if writerRecorder.Code != http.StatusNoContent || !nextCalled {
		t.Fatalf("feedback role status = %d, next called = %v", writerRecorder.Code, nextCalled)
	}
}

func TestAuthenticateRequestRestrictsLiveModelVerificationToOperators(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	reader := newTestUser(t, svc, "model-verify-reader@example.com", RoleRead, "active")
	writer := newTestUser(t, svc, "model-verify-writer@example.com", RoleWrite, "active")
	var calls int
	handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls++
		w.WriteHeader(http.StatusNoContent)
	}))

	readerResponse := httptest.NewRecorder()
	handler.ServeHTTP(
		readerResponse,
		newAuthenticatedRequest(t, svc, reader, http.MethodPost, "/api/models/verify", `{}`),
	)
	if readerResponse.Code != http.StatusForbidden || calls != 0 {
		t.Fatalf("reader status = %d, calls = %d", readerResponse.Code, calls)
	}

	writerResponse := httptest.NewRecorder()
	handler.ServeHTTP(
		writerResponse,
		newAuthenticatedRequest(t, svc, writer, http.MethodPost, "/api/models/verify", `{}`),
	)
	if writerResponse.Code != http.StatusNoContent || calls != 1 {
		t.Fatalf("writer status = %d, calls = %d", writerResponse.Code, calls)
	}
}
