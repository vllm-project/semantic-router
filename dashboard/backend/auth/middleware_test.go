package auth

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestPolicyMuxRequiresExplicitContracts(t *testing.T) {
	t.Parallel()

	routes := NewPolicyMux()
	routes.HandleFunc(PublicRoute("/api/auth/login", http.MethodPost), func(http.ResponseWriter, *http.Request) {})
	routes.HandleFunc(ProtectedRoute("/api/status", PermTopologyRead, SensitivityOperational, ResourceOwnerObservability, http.MethodGet), func(http.ResponseWriter, *http.Request) {})

	if policy, result := routes.LookupRoutePolicy(http.MethodPost, "/api/auth/login"); result != RouteFound || !policy.Public {
		t.Fatalf("public route lookup = (%+v, %v)", policy, result)
	}
	if policy, result := routes.LookupRoutePolicy(http.MethodGet, "/api/status"); result != RouteFound || policy.Permission != PermTopologyRead {
		t.Fatalf("protected route lookup = (%+v, %v)", policy, result)
	}
	if _, result := routes.LookupRoutePolicy(http.MethodGet, "/api/unknown"); result != RouteNotFound {
		t.Fatalf("unknown route lookup = %v, want %v", result, RouteNotFound)
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
			routes := NewPolicyMux()
			if strings.HasPrefix(tc.path, "/api/auth/login") || strings.HasPrefix(tc.path, "/api/setup/state") {
				routes.HandleFunc(PublicRoute(tc.path, http.MethodGet), func(http.ResponseWriter, *http.Request) {})
			} else if strings.HasPrefix(tc.path, "/api/") || strings.HasPrefix(tc.path, "/embedded/") {
				routes.HandleFunc(ProtectedRoute(tc.path, PermConfigRead, SensitivitySensitive, ResourceOwnerConfig, http.MethodGet), func(http.ResponseWriter, *http.Request) {})
			}
			handler := ServiceUnavailableGuard(routes)(next)

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

func TestAuthenticateRequestRequiresFeedbackPermissionForRouterOutcomes(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	reader := newTestUser(t, svc, "outcome-reader@example.com", RoleRead, "active")
	writer := newTestUser(t, svc, "outcome-writer@example.com", RoleWrite, "active")
	nextCalled := false
	handler := protectedTestHandler(svc, ProtectedRoute("/api/router/v1/router/outcomes", PermFeedbackSubmit, SensitivitySecret, ResourceOwnerFeedback, http.MethodPost), func(w http.ResponseWriter, _ *http.Request) {
		nextCalled = true
		w.WriteHeader(http.StatusNoContent)
	})

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

func TestAuthenticateRequestRequiresEvaluationRunForLiveModelVerification(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	reader := newTestUser(t, svc, "model-verify-reader@example.com", RoleRead, "active")
	writer := newTestUser(t, svc, "model-verify-writer@example.com", RoleWrite, "active")
	var calls int
	handler := protectedTestHandler(svc, ProtectedRoute("/api/models/verify", PermEvalRun, SensitivitySensitive, ResourceOwnerInference, http.MethodPost), func(w http.ResponseWriter, _ *http.Request) {
		calls++
		w.WriteHeader(http.StatusNoContent)
	})

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

func TestAuthenticateRequestDeniesUnknownProtectedRoutes(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "unknown-route@example.com", RoleAdmin, "active")
	routes := NewPolicyMux()
	nextCalled := false
	handler := AuthenticateRequest(svc, routes)(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		nextCalled = true
		w.WriteHeader(http.StatusNoContent)
	}))

	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, newAuthenticatedRequest(t, svc, user, http.MethodGet, "/api/not-registered", ""))

	if recorder.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusForbidden)
	}
	if nextCalled {
		t.Fatal("unknown protected route reached the handler")
	}
}

func TestInferencePermissionIsIndependentFromConfigRead(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	configOnly := newTestUser(t, svc, "config-only@example.com", RoleRead, "active")
	inferenceOnly := newTestUser(t, svc, "inference-only@example.com", RoleRead, "active")
	for _, denied := range []struct {
		user       *User
		permission string
	}{
		{user: configOnly, permission: PermInferenceRun},
		{user: inferenceOnly, permission: PermConfigRead},
	} {
		if _, err := svc.store.db.ExecContext(
			t.Context(),
			`INSERT INTO user_permissions(user_id, permission_key, allowed) VALUES(?,?,0)`,
			denied.user.ID,
			denied.permission,
		); err != nil {
			t.Fatalf("deny %s: %v", denied.permission, err)
		}
	}

	var calls int
	handler := protectedTestHandler(
		svc,
		ProtectedRoute("/api/router/v1/chat/completions", PermInferenceRun, SensitivitySecret, ResourceOwnerInference, http.MethodPost),
		func(w http.ResponseWriter, _ *http.Request) {
			calls++
			w.WriteHeader(http.StatusNoContent)
		},
	)

	deniedResponse := httptest.NewRecorder()
	handler.ServeHTTP(deniedResponse, newAuthenticatedRequest(t, svc, configOnly, http.MethodPost, "/api/router/v1/chat/completions", `{}`))
	if deniedResponse.Code != http.StatusForbidden || calls != 0 {
		t.Fatalf("config-only status = %d, calls = %d", deniedResponse.Code, calls)
	}

	allowedResponse := httptest.NewRecorder()
	handler.ServeHTTP(allowedResponse, newAuthenticatedRequest(t, svc, inferenceOnly, http.MethodPost, "/api/router/v1/chat/completions", `{}`))
	if allowedResponse.Code != http.StatusNoContent || calls != 1 {
		t.Fatalf("inference-only status = %d, calls = %d", allowedResponse.Code, calls)
	}
}

func TestIndependentPermissionGrantAndRevoke(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		method     string
		path       string
		permission string
		owner      ResourceOwner
	}{
		{name: "config", method: http.MethodPost, path: "/api/router/config/deploy", permission: PermConfigDeploy, owner: ResourceOwnerConfig},
		{name: "replay", method: http.MethodGet, path: "/api/router/v1/router_replay/replay-1", permission: PermReplayRead, owner: ResourceOwnerReplay},
		{name: "feedback", method: http.MethodPost, path: "/api/router/v1/router/outcomes", permission: PermFeedbackSubmit, owner: ResourceOwnerFeedback},
		{name: "inference", method: http.MethodPost, path: "/api/router/v1/chat/completions", permission: PermInferenceRun, owner: ResourceOwnerInference},
		{name: "ml", method: http.MethodPost, path: "/api/ml-pipeline/train", permission: PermMlPipeline, owner: ResourceOwnerML},
		{name: "openclaw", method: http.MethodPost, path: "/api/openclaw/provision", permission: PermOpenClaw, owner: ResourceOwnerOpenClaw},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			svc := newTestAuthService(t)
			role := "custom-" + test.name
			user := newTestCustomRoleUser(t, svc, test.name+"-custom@example.com", role)
			var calls int
			handler := protectedTestHandler(
				svc,
				ProtectedRoute(test.path, test.permission, SensitivitySensitive, test.owner, test.method),
				func(w http.ResponseWriter, _ *http.Request) {
					calls++
					w.WriteHeader(http.StatusNoContent)
				},
			)

			setRolePermission(t, svc, role, test.permission, true)
			allowed := httptest.NewRecorder()
			handler.ServeHTTP(allowed, newAuthenticatedRequest(t, svc, user, test.method, test.path, `{}`))
			if allowed.Code != http.StatusNoContent || calls != 1 {
				t.Fatalf("granted status = %d, calls = %d", allowed.Code, calls)
			}

			setRolePermission(t, svc, role, test.permission, false)
			denied := httptest.NewRecorder()
			handler.ServeHTTP(denied, newAuthenticatedRequest(t, svc, user, test.method, test.path, `{}`))
			if denied.Code != http.StatusForbidden || calls != 1 {
				t.Fatalf("revoked status = %d, calls = %d", denied.Code, calls)
			}
		})
	}
}

func newTestCustomRoleUser(t *testing.T, svc *Service, email, role string) *User {
	t.Helper()
	user := newTestUser(t, svc, email, RoleRead, "active")
	if _, err := svc.store.db.ExecContext(
		t.Context(),
		`UPDATE users SET role = ?, updated_at = ? WHERE id = ?`,
		role,
		nowUnix(),
		user.ID,
	); err != nil {
		t.Fatalf("assign custom role %q: %v", role, err)
	}
	user.Role = role
	return user
}

func setRolePermission(t *testing.T, svc *Service, role, permission string, allowed bool) {
	t.Helper()
	if _, err := svc.store.db.ExecContext(
		t.Context(),
		`INSERT INTO role_permissions(role, permission_key, allowed) VALUES(?,?,?)
		 ON CONFLICT(role, permission_key) DO UPDATE SET allowed = excluded.allowed`,
		role,
		permission,
		allowed,
	); err != nil {
		t.Fatalf("set role %s permission %s=%v: %v", role, permission, allowed, err)
	}
}

func setUserPermission(t *testing.T, svc *Service, userID, permission string, allowed bool) {
	t.Helper()
	if _, err := svc.store.db.ExecContext(
		t.Context(),
		`INSERT INTO user_permissions(user_id, permission_key, allowed) VALUES(?,?,?)
		 ON CONFLICT(user_id, permission_key) DO UPDATE SET allowed = excluded.allowed`,
		userID,
		permission,
		allowed,
	); err != nil {
		t.Fatalf("set user permission %s=%v: %v", permission, allowed, err)
	}
}

func TestMutationRequestBodyIsBounded(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "bounded-body@example.com", RoleAdmin, "active")
	var calls int
	handler := protectedTestHandler(
		svc,
		ProtectedMutationRoute("/api/config", PermConfigWrite, "config.update", SensitivitySecret, ResourceOwnerConfig, 4, http.MethodPost),
		func(w http.ResponseWriter, _ *http.Request) {
			calls++
			w.WriteHeader(http.StatusNoContent)
		},
	)

	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, newAuthenticatedRequest(t, svc, user, http.MethodPost, "/api/config", "12345"))
	if recorder.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusRequestEntityTooLarge)
	}
	if calls != 0 {
		t.Fatalf("handler calls = %d, want 0", calls)
	}
}

func TestBreakGlassAuthorizationIsTimeBounded(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "breakglass@example.com", RoleAdmin, "active")
	routes := NewPolicyMux()
	var calls int
	routes.HandleFunc(
		BreakGlassMutationRoute("/api/breakglass", "breakglass.apply", 64<<10, time.Nanosecond, http.MethodPost),
		func(w http.ResponseWriter, _ *http.Request) {
			calls++
			w.WriteHeader(http.StatusNoContent)
		},
	)
	handler := AuthenticateRequest(svc, routes)(routes)

	recorder := httptest.NewRecorder()
	handler.ServeHTTP(recorder, newAuthenticatedRequest(t, svc, user, http.MethodPost, "/api/breakglass", `{}`))
	if recorder.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusForbidden)
	}
	if calls != 0 {
		t.Fatalf("handler calls = %d, want 0", calls)
	}
}

func TestRejectRevokedMutationRequiresLivePermission(t *testing.T) {
	t.Parallel()

	recorder := httptest.NewRecorder()
	plain := httptest.NewRequest(http.MethodPost, "/api/router/config/update", nil)
	if RejectRevokedMutation(recorder, plain) {
		t.Fatal("missing revalidator should not reject unit-test handler calls")
	}

	denied := plain.WithContext(WithPermissionRevalidator(plain.Context(), func(context.Context) error {
		return errPermissionDenied
	}))
	if !RejectRevokedMutation(recorder, denied) {
		t.Fatal("live revalidation failure should reject the mutation")
	}
	if recorder.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusForbidden)
	}
}

func TestMutationRechecksPermissionAfterPausedBody(t *testing.T) {
	t.Parallel()

	svc := newTestAuthService(t)
	user := newTestUser(t, svc, "paused-body@example.com", RoleAdmin, "active")
	token, err := svc.issueToken(user)
	if err != nil {
		t.Fatalf("issue token: %v", err)
	}

	body := &pausingRequestBody{
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
	req := httptest.NewRequest(http.MethodPost, "/api/config", body)
	req.Header.Set("Authorization", "Bearer "+token)

	var calls int
	handler := protectedTestHandler(
		svc,
		ProtectedMutationRoute("/api/config", PermConfigWrite, "config.update", SensitivitySecret, ResourceOwnerConfig, 64, http.MethodPost),
		func(w http.ResponseWriter, _ *http.Request) {
			calls++
			w.WriteHeader(http.StatusNoContent)
		},
	)
	recorder := httptest.NewRecorder()
	done := make(chan struct{})
	go func() {
		handler.ServeHTTP(recorder, req)
		close(done)
	}()

	select {
	case <-body.started:
	case <-time.After(5 * time.Second):
		t.Fatal("middleware did not start reading the request body")
	}
	if err := svc.RevokeToken(t.Context(), token); err != nil {
		t.Fatalf("revoke token: %v", err)
	}
	close(body.release)

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("request did not finish")
	}
	if recorder.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusForbidden)
	}
	if calls != 0 {
		t.Fatalf("handler calls = %d, want 0", calls)
	}
}

type pausingRequestBody struct {
	started chan struct{}
	release chan struct{}
	sent    bool
}

func (b *pausingRequestBody) Read(payload []byte) (int, error) {
	if !b.sent {
		b.sent = true
		close(b.started)
		return copy(payload, []byte(`{}`)), nil
	}
	<-b.release
	return 0, io.EOF
}

func (b *pausingRequestBody) Close() error {
	return nil
}
