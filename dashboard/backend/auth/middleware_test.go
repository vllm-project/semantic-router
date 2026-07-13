package auth

import (
	"bufio"
	"errors"
	"io"
	"net"
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

func TestAuthenticateRequestDisablesCachingForEveryProtectedOutcome(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "protected-cache@example.com", RoleAdmin, "active")
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}
	readUser := newTestUser(t, svc, "protected-cache-read@example.com", RoleRead, "active")
	readToken, err := svc.issueToken(readUser)
	if err != nil {
		t.Fatalf("issue read token: %v", err)
	}

	testCases := []struct {
		name       string
		path       string
		token      string
		nextCode   int
		wantCode   int
		wantCalled bool
		flush      bool
	}{
		{name: "success", path: "/api/status", token: token, nextCode: http.StatusOK, wantCode: http.StatusOK, wantCalled: true},
		{name: "handler error", path: "/api/status", token: token, nextCode: http.StatusInternalServerError, wantCode: http.StatusInternalServerError, wantCalled: true},
		{name: "stream", path: "/api/status", token: token, nextCode: http.StatusOK, wantCode: http.StatusOK, wantCalled: true, flush: true},
		{name: "forbidden", path: "/api/admin/permissions", token: readToken, wantCode: http.StatusForbidden},
		{name: "missing token", path: "/api/status", wantCode: http.StatusUnauthorized},
		{name: "invalid token", path: "/api/status", token: "not-a-token", wantCode: http.StatusUnauthorized},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			nextCalled := false
			handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				nextCalled = true
				w.WriteHeader(tc.nextCode)
				if tc.flush {
					w.(http.Flusher).Flush()
				}
			}))
			req := httptest.NewRequest(http.MethodGet, tc.path, nil)
			if tc.token != "" {
				req.Header.Set("Authorization", "Bearer "+tc.token)
			}
			rec := httptest.NewRecorder()

			handler.ServeHTTP(rec, req)

			if rec.Code != tc.wantCode {
				t.Fatalf("status = %d, want %d", rec.Code, tc.wantCode)
			}
			if nextCalled != tc.wantCalled {
				t.Fatalf("next called = %v, want %v", nextCalled, tc.wantCalled)
			}
			if got := rec.Header().Get("Cache-Control"); got != "no-store" {
				t.Fatalf("Cache-Control = %q, want no-store", got)
			}
			if got := rec.Header().Get("Pragma"); got != "no-cache" {
				t.Fatalf("Pragma = %q, want no-cache", got)
			}
		})
	}
}

func TestAuthenticateRequestLeavesPublicStaticCachePolicyUntouched(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
		w.WriteHeader(http.StatusOK)
	}))
	req := httptest.NewRequest(http.MethodGet, "/assets/app.01234567.js", nil)
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if got := rec.Header().Get("Cache-Control"); got != "public, max-age=31536000, immutable" {
		t.Fatalf("Cache-Control = %q, want immutable static policy", got)
	}
}

func TestAuthenticateRequestEnforcesNoStoreAfterHandlerOverwritesCachePolicy(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "protected-overwrite@example.com", RoleAdmin, "active")
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}

	testCases := []struct {
		name    string
		handler http.HandlerFunc
	}{
		{
			name: "explicit WriteHeader",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Cache-Control", "public, max-age=3600")
				w.Header().Set("Pragma", "cache")
				w.WriteHeader(http.StatusNoContent)
			},
		},
		{
			name: "implicit WriteHeader through Write",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Cache-Control", "no-cache")
				_, _ = io.WriteString(w, "protected")
			},
		},
		{
			name: "header-only return",
			handler: func(w http.ResponseWriter, _ *http.Request) {
				w.Header().Set("Cache-Control", "public")
				w.Header().Set("Pragma", "cache")
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			handler := AuthenticateRequest(svc)(tc.handler)
			req := httptest.NewRequest(http.MethodGet, "/api/status", nil)
			req.Header.Set("Authorization", "Bearer "+token)
			recorder := httptest.NewRecorder()

			handler.ServeHTTP(recorder, req)

			if got := recorder.Header().Get("Cache-Control"); got != "no-store" {
				t.Fatalf("Cache-Control = %q, want no-store", got)
			}
			if got := recorder.Header().Get("Pragma"); got != "no-cache" {
				t.Fatalf("Pragma = %q, want no-cache", got)
			}
		})
	}
}

func TestAuthenticateRequestRejectsUntrustedProtectedWebSocketOrigins(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "websocket-origin@example.com", RoleAdmin, "active")
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}

	testCases := []struct {
		name   string
		path   string
		origin string
	}{
		{name: "room sibling origin", path: "/api/openclaw/rooms/room/ws", origin: "http://evil.example.com"},
		{name: "embedded proxy sibling origin", path: "/embedded/openclaw/worker/socket", origin: "http://evil.example.com"},
		{name: "room missing origin", path: "/api/openclaw/rooms/room/ws"},
		{name: "embedded proxy missing origin", path: "/embedded/openclaw/worker/socket"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			nextCalled := false
			handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				nextCalled = true
				w.WriteHeader(http.StatusNoContent)
			}))
			request := httptest.NewRequest(http.MethodGet, "http://play.example.com"+tc.path, nil)
			request.Host = "play.example.com"
			request.Header.Set("Authorization", "Bearer "+token)
			request.Header.Set("Connection", "keep-alive, Upgrade")
			request.Header.Set("Upgrade", "websocket")
			if tc.origin != "" {
				request.Header.Set("Origin", tc.origin)
			}
			recorder := httptest.NewRecorder()

			handler.ServeHTTP(recorder, request)

			if recorder.Code != http.StatusForbidden {
				t.Fatalf("status = %d, want %d", recorder.Code, http.StatusForbidden)
			}
			if nextCalled {
				t.Fatal("untrusted WebSocket request reached the protected handler")
			}
			if got := recorder.Header().Get("Cache-Control"); got != "no-store" {
				t.Fatalf("Cache-Control = %q, want no-store", got)
			}
		})
	}
}

func TestAuthenticateRequestAllowsSameOriginProtectedWebSocket(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "same-origin-websocket@example.com", RoleAdmin, "active")
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issueToken() error = %v", err)
	}
	nextCalled := false
	handler := AuthenticateRequest(svc)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		nextCalled = true
		w.WriteHeader(http.StatusNoContent)
	}))
	request := httptest.NewRequest(http.MethodGet, "http://internal:8700/embedded/openclaw/worker/socket", nil)
	request.Host = "play.example.com"
	request.Header.Set("Authorization", "Bearer "+token)
	request.Header.Set("Connection", "Upgrade")
	request.Header.Set("Upgrade", "websocket")
	request.Header.Set("Origin", "https://play.example.com")
	request.Header.Set(forwardedProtoHeader, "https")
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, request)

	if recorder.Code != http.StatusNoContent || !nextCalled {
		t.Fatalf("same-origin WebSocket result = status %d, next %v", recorder.Code, nextCalled)
	}
}

func TestProtectedResponseWriterPreservesStreamingAndUpgradeInterfaces(t *testing.T) {
	t.Parallel()

	underlying := newOptionalResponseWriter()
	wrapped := newProtectedResponseWriter(underlying)
	wrapped.Header().Set("Cache-Control", "public")
	wrapped.Header().Set("Pragma", "cache")

	if unwrapper, ok := any(wrapped).(interface{ Unwrap() http.ResponseWriter }); !ok {
		t.Fatal("protected writer does not expose Unwrap")
	} else if unwrapper.Unwrap() != underlying {
		t.Fatal("Unwrap did not return the underlying writer")
	}
	hijacker, ok := any(wrapped).(http.Hijacker)
	if !ok {
		t.Fatal("protected writer does not preserve http.Hijacker")
	}
	if _, _, err := hijacker.Hijack(); !errors.Is(err, http.ErrNotSupported) {
		t.Fatalf("Hijack() error = %v, want http.ErrNotSupported", err)
	}
	if !underlying.hijacked {
		t.Fatal("Hijack was not forwarded to the underlying writer")
	}
	pusher, ok := any(wrapped).(http.Pusher)
	if !ok {
		t.Fatal("protected writer does not preserve http.Pusher")
	}
	if err := pusher.Push("/asset.js", nil); !errors.Is(err, http.ErrNotSupported) {
		t.Fatalf("Push() error = %v, want http.ErrNotSupported", err)
	}
	if !underlying.pushed {
		t.Fatal("Push was not forwarded to the underlying writer")
	}
	readerFrom, ok := any(wrapped).(io.ReaderFrom)
	if !ok {
		t.Fatal("protected writer does not preserve io.ReaderFrom")
	}
	if _, err := readerFrom.ReadFrom(strings.NewReader("streamed")); err != nil {
		t.Fatalf("ReadFrom() error = %v", err)
	}
	if got := underlying.body.String(); got != "streamed" {
		t.Fatalf("streamed body = %q, want streamed", got)
	}

	wrapped.Header().Set("Cache-Control", "no-cache")
	wrapped.Header().Set("Pragma", "cache")
	if err := http.NewResponseController(wrapped).Flush(); err != nil {
		t.Fatalf("ResponseController.Flush() error = %v", err)
	}
	if !underlying.flushed {
		t.Fatal("Flush was not forwarded to the underlying writer")
	}
	if got := underlying.Header().Get("Cache-Control"); got != "no-store" {
		t.Fatalf("Cache-Control after streaming = %q, want no-store", got)
	}
	if got := underlying.Header().Get("Pragma"); got != "no-cache" {
		t.Fatalf("Pragma after streaming = %q, want no-cache", got)
	}
}

type optionalResponseWriter struct {
	header   http.Header
	body     strings.Builder
	status   int
	flushed  bool
	hijacked bool
	pushed   bool
}

func newOptionalResponseWriter() *optionalResponseWriter {
	return &optionalResponseWriter{header: make(http.Header)}
}

func (w *optionalResponseWriter) Header() http.Header { return w.header }

func (w *optionalResponseWriter) Write(data []byte) (int, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	return w.body.Write(data)
}

func (w *optionalResponseWriter) WriteHeader(statusCode int) { w.status = statusCode }

func (w *optionalResponseWriter) ReadFrom(reader io.Reader) (int64, error) {
	if w.status == 0 {
		w.status = http.StatusOK
	}
	return io.Copy(&w.body, reader)
}

func (w *optionalResponseWriter) Flush() { w.flushed = true }

func (w *optionalResponseWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	w.hijacked = true
	return nil, nil, http.ErrNotSupported
}

func (w *optionalResponseWriter) Push(string, *http.PushOptions) error {
	w.pushed = true
	return http.ErrNotSupported
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
		{method: http.MethodPost, path: "/api/auth/password", expected: ""},
		{method: http.MethodPatch, path: "/api/admin/users/user-1", expected: PermUsersManage},
		{method: http.MethodGet, path: "/api/admin/audit-logs", expected: PermUsersManage},
		{method: http.MethodGet, path: "/api/status", expected: PermLogsRead},
		{method: http.MethodGet, path: "/embedded/grafana/", expected: PermLogsRead},
		{method: http.MethodGet, path: "/embedded/wizmap/", expected: PermConfigRead},
		{method: http.MethodPost, path: "/api/setup/activate", expected: PermConfigWrite},
		{method: http.MethodPost, path: "/api/setup/import-remote", expected: PermConfigWrite},
		{method: http.MethodGet, path: "/api/mcp/servers", expected: PermMcpRead},
		{method: http.MethodPost, path: "/api/mcp/servers", expected: PermMcpManage},
		{method: http.MethodDelete, path: "/api/mcp/servers/server-1/status", expected: PermMcpManage},
		{method: http.MethodPost, path: "/api/mcp/servers/server-1/connect", expected: PermMcpManage},
		{method: http.MethodPost, path: "/api/router/config/deploy", expected: PermConfigDeploy},
		{method: http.MethodPost, path: "/api/router/config/deploy/preview", expected: PermConfigDeploy},
		{method: http.MethodGet, path: "/api/router/config/deployments", expected: PermConfigRead},
		{method: http.MethodPost, path: "/api/evaluation/tasks", expected: PermEvalWrite},
		{method: http.MethodPost, path: "/api/evaluation/run", expected: PermEvalRun},
		{method: http.MethodPost, path: "/api/evaluation/cancel/task-1", expected: PermEvalRun},
		{method: http.MethodGet, path: "/api/fleet-sim/api/workloads", expected: PermConfigRead},
		{method: http.MethodPost, path: "/api/fleet-sim/api/jobs", expected: PermConfigWrite},
		{method: http.MethodGet, path: "/api/openclaw/teams", expected: PermOpenClawRead},
		{method: http.MethodPost, path: "/api/openclaw/teams", expected: PermOpenClaw},
		{method: http.MethodPost, path: "/api/openclaw/rooms/room-1/messages", expected: PermOpenClawRead},
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
